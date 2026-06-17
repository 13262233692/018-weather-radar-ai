"""
GPU 显存管理器 - 监控、碎片整理、安全回收

核心问题:
  PyTorch Caching Allocator 将已释放的显存块保留在缓存中，不同尺寸的张量
  分配后会产生大量碎片，导致总空闲显存足够但无法分配连续大块，触发 OOM。

解决方案:
  1. 推理前后主动调用 empty_cache() 释放碎片化缓存
  2. 监控显存水位，超阈值时拒绝新请求（背压）
  3. 推理前预分配最大 batch 所需显存，确保不会在中途 OOM
  4. 使用 PYTORCH_CUDA_ALLOC_CONF 环境变量配置分配策略
"""
import os
import logging
from typing import Optional
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)


@dataclass
class GPUMemoryStats:
    allocated_mb: float = 0.0
    reserved_mb: float = 0.0
    free_mb: float = 0.0
    total_mb: float = 0.0
    fragmentation_ratio: float = 0.0

    @property
    def utilization(self) -> float:
        if self.total_mb <= 0:
            return 0.0
        return self.allocated_mb / self.total_mb


class GPUMemoryManager:
    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        gpu_cfg = self.config.get("gpu", {})

        self.device = torch.device(
            gpu_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )
        self.high_watermark_ratio = gpu_cfg.get("high_watermark_ratio", 0.85)
        self.low_watermark_ratio = gpu_cfg.get("low_watermark_ratio", 0.60)
        self.enable_memory_pool = gpu_cfg.get("enable_memory_pool", True)
        self.max_batch_memory_mb = gpu_cfg.get("max_batch_memory_mb", 4096)

        self._is_cuda = self.device.type == "cuda"
        self._memory_reserved = False

        if self._is_cuda and self.enable_memory_pool:
            self._configure_allocator()

    def _configure_allocator(self):
        existing = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
        opts = []
        if "max_split_size_mb" not in existing:
            opts.append("max_split_size_mb:128")
        if "garbage_collection_threshold" not in existing:
            opts.append("garbage_collection_threshold:0.6")
        if opts:
            new_conf = ",".join(opts)
            if existing:
                new_conf = f"{existing},{new_conf}"
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = new_conf
            logger.info(f"CUDA allocator configured: {new_conf}")

    def get_stats(self) -> GPUMemoryStats:
        if not self._is_cuda:
            return GPUMemoryStats(
                allocated_mb=0.0,
                reserved_mb=0.0,
                free_mb=float("inf"),
                total_mb=float("inf"),
                fragmentation_ratio=0.0,
            )

        allocated = torch.cuda.memory_allocated(self.device) / (1024 * 1024)
        reserved = torch.cuda.memory_reserved(self.device) / (1024 * 1024)
        total = torch.cuda.get_device_properties(self.device).total_mem / (1024 * 1024)
        free = total - allocated

        fragmentation = 0.0
        if reserved > 0:
            fragmentation = (reserved - allocated) / reserved

        return GPUMemoryStats(
            allocated_mb=allocated,
            reserved_mb=reserved,
            free_mb=free,
            total_mb=total,
            fragmentation_ratio=fragmentation,
        )

    def can_allocate(self, required_mb: float) -> bool:
        stats = self.get_stats()
        if not self._is_cuda:
            return True
        effective_free = stats.free_mb
        return effective_free >= required_mb

    def is_under_pressure(self) -> bool:
        stats = self.get_stats()
        if not self._is_cuda:
            return False
        if stats.utilization > self.high_watermark_ratio:
            return True
        if stats.fragmentation_ratio > 0.5:
            return True
        return False

    def pre_inference_cleanup(self):
        if not self._is_cuda:
            return
        self._synchronize()
        self._empty_cache()
        stats = self.get_stats()
        logger.debug(
            f"Pre-inference GPU: "
            f"alloc={stats.allocated_mb:.0f}MB, "
            f"reserved={stats.reserved_mb:.0f}MB, "
            f"free={stats.free_mb:.0f}MB, "
            f"frag={stats.fragmentation_ratio:.2%}"
        )

    def post_inference_cleanup(self):
        if not self._is_cuda:
            return
        self._synchronize()
        self._empty_cache()
        stats = self.get_stats()
        logger.debug(
            f"Post-inference GPU: "
            f"alloc={stats.allocated_mb:.0f}MB, "
            f"reserved={stats.reserved_mb:.0f}MB, "
            f"free={stats.free_mb:.0f}MB, "
            f"frag={stats.fragmentation_ratio:.2%}"
        )

    def estimate_batch_memory_mb(
        self,
        batch_size: int,
        seq_len: int,
        channels: int,
        height: int,
        width: int,
        num_layers: int = 3,
    ) -> float:
        element_bytes = 4
        input_size = batch_size * seq_len * channels * height * width * element_bytes
        hidden_size = batch_size * num_layers * 2 * 64 * height * width * element_bytes
        output_size = batch_size * 12 * channels * height * width * element_bytes
        intermediate_factor = 3.0
        total = (input_size + hidden_size + output_size) * intermediate_factor
        return total / (1024 * 1024)

    def safe_inference_context(self):
        return _SafeInferenceContext(self)

    def _synchronize(self):
        if self._is_cuda:
            try:
                torch.cuda.synchronize(self.device)
            except RuntimeError:
                pass

    def _empty_cache(self):
        if self._is_cuda:
            try:
                torch.cuda.empty_cache()
            except RuntimeError:
                pass


class _SafeInferenceContext:
    def __init__(self, manager: GPUMemoryManager):
        self.manager = manager

    def __enter__(self):
        self.manager.pre_inference_cleanup()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.manager.post_inference_cleanup()
        return False
