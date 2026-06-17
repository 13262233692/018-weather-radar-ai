"""
Dynamic Batching 调度器 - 异步请求队列与智能批处理

重构支持双模态输入:
  - 同时接收雷达张量 + 卫星张量 + 硬注意力掩码
  - 自动检测是否启用卫星分支
  - 批处理时分别对两种模态进行拼接
"""
import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

import torch

from .gpu_manager import GPUMemoryManager
from ..preprocessing.tensor_standardizer import TensorStandardizer
from ..models.multimodal_fusion import MultimodalConvLSTM

logger = logging.getLogger(__name__)


@dataclass
class InferenceRequest:
    request_id: str
    radar_tensor: torch.Tensor
    satellite_tensor: Optional[torch.Tensor] = None
    hard_mask_tensor: Optional[torch.Tensor] = None
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

        self.use_satellite = isinstance(model, MultimodalConvLSTM) and getattr(
            model, "use_satellite", True
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
            "multimodal_requests": 0,
            "radar_only_requests": 0,
        }

    async def start(self):
        if self._running:
            return
        self._running = True
        self._worker_task = asyncio.create_task(self._batch_worker())
        logger.info(
            f"BatchScheduler started: max_batch={self.max_batch_size}, "
            f"max_wait={self.max_wait_ms}ms, "
            f"multimodal={'enabled' if self.use_satellite else 'disabled'}"
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

    async def submit(
        self,
        radar_tensor: torch.Tensor,
        satellite_tensor: Optional[torch.Tensor] = None,
        hard_mask_tensor: Optional[torch.Tensor] = None,
        meta: Dict = None,
    ) -> torch.Tensor:
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

        has_satellite = satellite_tensor is not None
        if has_satellite:
            self._stats["multimodal_requests"] += 1
        else:
            self._stats["radar_only_requests"] += 1

        loop = asyncio.get_event_loop()
        future = loop.create_future()

        request = InferenceRequest(
            request_id=request_id,
            radar_tensor=radar_tensor,
            satellite_tensor=satellite_tensor,
            hard_mask_tensor=hard_mask_tensor,
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
            first = await asyncio.wait_for(self._queue.get(), timeout=1.0)
        except asyncio.TimeoutError:
            return []

        batch = [first]
        deadline = time.monotonic() + self.max_wait_ms / 1000.0

        while len(batch) < self.max_batch_size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                req = await asyncio.wait_for(self._queue.get(), timeout=remaining)
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

        try:
            batched_radar, batched_satellite, batched_mask, metas, has_satellite_batch = (
                self._prepare_batch_tensors(batch)
            )
        except RuntimeError as e:
            logger.error(f"Batch tensor preparation failed: {e}")
            for req in batch:
                if not req.future.done():
                    req.future.set_exception(e)
            return

        estimated_mb = self.gpu_manager.estimate_batch_memory_mb(
            batch_size=batched_radar.size(0),
            seq_len=batched_radar.size(1),
            channels=batched_radar.size(2),
            height=batched_radar.size(3),
            width=batched_radar.size(4),
            num_layers=len(self.model.encoder_layers),
        )

        if has_satellite_batch and batched_satellite is not None:
            estimated_mb *= 1.2

        if not self.gpu_manager.can_allocate(estimated_mb):
            self._stats["oom_count"] += 1
            for i, req in enumerate(batch):
                if not req.future.done():
                    if i == 0:
                        try:
                            result = await self._fallback_single_execute(req)
                            req.future.set_result(result)
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
                batched_radar = batched_radar.to(self.device)
                if batched_satellite is not None:
                    batched_satellite = batched_satellite.to(self.device)
                if batched_mask is not None:
                    batched_mask = batched_mask.to(self.device)

                with torch.no_grad():
                    self.model.eval()
                    if has_satellite_batch and self.use_satellite:
                        batched_output = self.model(
                            batched_radar,
                            batched_satellite,
                            batched_mask,
                        )
                    else:
                        batched_output = self.model(batched_radar)

                batched_output = batched_output.cpu()

            for i, req in enumerate(batch):
                if not req.future.done():
                    single_output = batched_output[i : i + 1]
                    restored = self.standardizer.restore_tensor(single_output, metas[i])
                    req.future.set_result(restored)

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

    def _prepare_batch_tensors(
        self, batch: List[InferenceRequest]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], List[Dict], bool]:
        radar_tensors = []
        satellite_tensors = []
        hard_mask_tensors = []
        metas = []

        has_satellite = any(req.satellite_tensor is not None for req in batch)

        for req in batch:
            std_radar, meta = self.standardizer.standardize_tensor(req.radar_tensor)
            radar_tensors.append(std_radar)
            metas.append(meta)

            if has_satellite:
                if req.satellite_tensor is not None:
                    std_sat, _ = self.standardizer.standardize_tensor(req.satellite_tensor)
                    satellite_tensors.append(std_sat)
                else:
                    placeholder = torch.zeros_like(std_radar[:, :, :2, :, :])
                    satellite_tensors.append(placeholder)

                if req.hard_mask_tensor is not None:
                    std_mask, _ = self.standardizer.standardize_tensor(req.hard_mask_tensor)
                    hard_mask_tensors.append(std_mask)
                else:
                    hard_mask_tensors.append(None)

        try:
            batched_radar = torch.cat(radar_tensors, dim=0)
        except RuntimeError as e:
            raise RuntimeError(f"Radar tensor concatenation failed: {e}")

        batched_satellite = None
        batched_mask = None

        if has_satellite:
            try:
                batched_satellite = torch.cat(satellite_tensors, dim=0)
            except RuntimeError as e:
                raise RuntimeError(f"Satellite tensor concatenation failed: {e}")

            valid_masks = [m for m in hard_mask_tensors if m is not None]
            if valid_masks:
                try:
                    batched_mask = torch.cat(valid_masks, dim=0)
                except RuntimeError as e:
                    logger.warning(f"Hard mask concatenation failed (continuing without): {e}")
                    batched_mask = None

        return batched_radar, batched_satellite, batched_mask, metas, has_satellite

    async def _fallback_single_execute(
        self, req: InferenceRequest
    ) -> torch.Tensor:
        self.gpu_manager.pre_inference_cleanup()

        radar = req.radar_tensor.to(self.device)
        satellite = req.satellite_tensor.to(self.device) if req.satellite_tensor is not None else None
        mask = req.hard_mask_tensor.to(self.device) if req.hard_mask_tensor is not None else None

        with torch.no_grad():
            self.model.eval()
            if satellite is not None and self.use_satellite:
                output = self.model(radar, satellite, mask)
            else:
                output = self.model(radar)

        output = output.cpu()
        self.gpu_manager.post_inference_cleanup()
        return output

    def get_stats(self) -> Dict[str, Any]:
        gpu_stats = self.gpu_manager.get_stats()
        return {
            **self._stats,
            "queue_size": self._queue.qsize(),
            "multimodal_enabled": self.use_satellite,
            "gpu_allocated_mb": gpu_stats.allocated_mb,
            "gpu_free_mb": gpu_stats.free_mb,
            "gpu_fragmentation": gpu_stats.fragmentation_ratio,
            "gpu_utilization": gpu_stats.utilization,
        }
