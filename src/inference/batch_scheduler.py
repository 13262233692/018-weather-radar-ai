"""
Dynamic Batching 调度器 - 异步请求队列与智能批处理

核心问题:
  多个 Uvicorn Worker 同时发起推理请求，每个请求单独创建 GPU 张量并执行推理，
  导致:
  1. 并发 GPU 访问没有序列化，多个 kernel 同时竞争显存
  2. 无法利用 batch 维度合并多个小请求，GPU 利用率低
  3. 请求完成后张量未被及时释放，碎片累积

解决方案:
  1. 所有推理请求通过 asyncio 队列提交，由单一推理 Worker 消费
  2. 推理 Worker 在等待窗口内收集多个请求，合并为一个 batch 执行
  3. 推理完成后将结果拆分回各请求的 Future
  4. GPU 显存操作全部序列化在单一线程，消除并发踩踏
"""
import asyncio
import logging
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

import torch

from .gpu_manager import GPUMemoryManager
from ..preprocessing.tensor_standardizer import TensorStandardizer

logger = logging.getLogger(__name__)


@dataclass
class InferenceRequest:
    request_id: str
    input_tensor: torch.Tensor
    meta: Dict = field(default_factory=dict)
    future: asyncio.Future = field(default=None)
    submit_time: float = field(default_factory=time.monotonic)


class DynamicBatchScheduler:
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        config: Optional[dict] = None,
    ):
        self.config = config or {}
        batch_cfg = self.config.get("batch_scheduler", {})

        self.model = model
        self.device = device
        self.gpu_manager = GPUMemoryManager(self.config)
        self.standardizer = TensorStandardizer(
            target_height=self.config.get("model", {}).get("img_height", 256),
            target_width=self.config.get("model", {}).get("img_width", 256),
        )

        self.max_batch_size = batch_cfg.get("max_batch_size", 8)
        self.max_wait_ms = batch_cfg.get("max_wait_ms", 50)
        self.max_queue_size = batch_cfg.get("max_queue_size", 64)

        self._queue: asyncio.Queue = asyncio.Queue(maxsize=self.max_queue_size)
        self._running = False
        self._worker_task: Optional[asyncio.Task] = None
        self._request_counter = 0
        self._lock = asyncio.Lock()

        self._stats = {
            "total_requests": 0,
            "total_batches": 0,
            "avg_batch_size": 0.0,
            "avg_wait_ms": 0.0,
            "total_inference_ms": 0.0,
            "oom_count": 0,
            "dropped_count": 0,
        }

    async def start(self):
        if self._running:
            return
        self._running = True
        self._worker_task = asyncio.create_task(self._batch_worker())
        logger.info(
            f"BatchScheduler started: max_batch={self.max_batch_size}, "
            f"max_wait={self.max_wait_ms}ms"
        )

    async def stop(self):
        self._running = False
        if self._worker_task is not None:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        logger.info("BatchScheduler stopped")

    async def submit(self, input_tensor: torch.Tensor, meta: Dict = None) -> torch.Tensor:
        if self._queue.full():
            self._stats["dropped_count"] += 1
            raise RuntimeError("Inference queue full - too many concurrent requests")

        if self.gpu_manager.is_under_pressure():
            self.gpu_manager.post_inference_cleanup()
            if self.gpu_manager.is_under_pressure():
                raise RuntimeError("GPU under memory pressure - retry later")

        self._request_counter += 1
        request_id = f"req_{self._request_counter}"
        meta = meta or {}

        loop = asyncio.get_event_loop()
        future = loop.create_future()

        request = InferenceRequest(
            request_id=request_id,
            input_tensor=input_tensor,
            meta=meta,
            future=future,
        )

        await self._queue.put(request)
        self._stats["total_requests"] += 1

        return await future

    async def _batch_worker(self):
        while self._running:
            try:
                batch = await self._collect_batch()
                if not batch:
                    continue
                await self._execute_batch(batch)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Batch worker error: {e}", exc_info=True)
                await asyncio.sleep(0.1)

    async def _collect_batch(self) -> List[InferenceRequest]:
        try:
            first = await asyncio.wait_for(
                self._queue.get(), timeout=1.0
            )
        except asyncio.TimeoutError:
            return []

        batch = [first]
        deadline = time.monotonic() + self.max_wait_ms / 1000.0

        while len(batch) < self.max_batch_size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                req = await asyncio.wait_for(
                    self._queue.get(), timeout=remaining
                )
                batch.append(req)
            except asyncio.TimeoutError:
                break

        return batch

    async def _execute_batch(self, batch: List[InferenceRequest]):
        batch_size = len(batch)
        self._stats["total_batches"] += 1

        wait_times = [time.monotonic() - r.submit_time for r in batch]
        avg_wait = sum(wait_times) / len(wait_times) * 1000
        self._stats["avg_wait_ms"] = (
            (self._stats["avg_wait_ms"] * (self._stats["total_batches"] - 1) + avg_wait)
            / self._stats["total_batches"]
        )
        self._stats["avg_batch_size"] = (
            (self._stats["avg_batch_size"] * (self._stats["total_batches"] - 1) + batch_size)
            / self._stats["total_batches"]
        )

        logger.debug(
            f"Executing batch: size={batch_size}, "
            f"avg_wait={avg_wait:.1f}ms"
        )

        standardized_tensors = []
        metas = []

        for req in batch:
            std_tensor, meta = self.standardizer.standardize_tensor(req.input_tensor)
            standardized_tensors.append(std_tensor)
            metas.append(meta)

        try:
            batched_input = torch.cat(standardized_tensors, dim=0)
        except RuntimeError as e:
            logger.error(f"Tensor concatenation failed: {e}")
            for req in batch:
                if not req.future.done():
                    req.future.set_exception(e)
            return

        estimated_mb = self.gpu_manager.estimate_batch_memory_mb(
            batch_size=batched_input.size(0),
            seq_len=batched_input.size(1),
            channels=batched_input.size(2),
            height=batched_input.size(3),
            width=batched_input.size(4),
            num_layers=len(self.model.encoder_layers),
        )

        if not self.gpu_manager.can_allocate(estimated_mb):
            self._stats["oom_count"] += 1
            for i, req in enumerate(batch):
                if not req.future.done():
                    if i == 0:
                        try:
                            result = await self._fallback_single_execute(
                                standardized_tensors[0]
                            )
                            restored = self.standardizer.restore_tensor(result, metas[0])
                            req.future.set_result(restored)
                        except Exception as exc:
                            req.future.set_exception(exc)
                    else:
                        req.future.set_exception(
                            RuntimeError(
                                f"GPU OOM - batch dropped. "
                                f"Required {estimated_mb:.0f}MB"
                            )
                        )
            self.gpu_manager.post_inference_cleanup()
            return

        try:
            with self.gpu_manager.safe_inference_context():
                batched_input = batched_input.to(self.device)

                with torch.no_grad():
                    self.model.eval()
                    batched_output = self.model(batched_input)

                batched_output = batched_output.cpu()

            for i, req in enumerate(batch):
                if not req.future.done():
                    single_output = batched_output[i : i + 1]
                    restored = self.standardizer.restore_tensor(single_output, metas[i])
                    req.future.set_result(restored)

            inference_ms = 0.0
            self._stats["total_inference_ms"] += inference_ms

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                self._stats["oom_count"] += 1
                logger.error(
                    f"CUDA OOM during batch inference "
                    f"(batch_size={batch_size}, estimated={estimated_mb:.0f}MB)"
                )
                self.gpu_manager.post_inference_cleanup()

                for req in batch:
                    if not req.future.done():
                        req.future.set_exception(
                            RuntimeError("CUDA out of memory - request dropped")
                        )
            else:
                for req in batch:
                    if not req.future.done():
                        req.future.set_exception(e)

    async def _fallback_single_execute(
        self, tensor: torch.Tensor
    ) -> torch.Tensor:
        self.gpu_manager.pre_inference_cleanup()
        tensor = tensor.to(self.device)
        with torch.no_grad():
            self.model.eval()
            output = self.model(tensor)
        output = output.cpu()
        self.gpu_manager.post_inference_cleanup()
        return output

    def get_stats(self) -> Dict[str, Any]:
        gpu_stats = self.gpu_manager.get_stats()
        return {
            **self._stats,
            "queue_size": self._queue.qsize(),
            "gpu_allocated_mb": gpu_stats.allocated_mb,
            "gpu_free_mb": gpu_stats.free_mb,
            "gpu_fragmentation": gpu_stats.fragmentation_ratio,
            "gpu_utilization": gpu_stats.utilization,
        }
