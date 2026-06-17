"""
张量尺寸标准化器 - 解决不同仰角雷达快照因边界裁剪导致的尺寸差异

核心问题:
  不同仰角的径向数据在极坐标→笛卡尔转换后，实际有效区域大小不一，
  导致进 GPU 的张量尺寸不统一，PyTorch Caching Allocator 为每种尺寸
  分配独立显存块，反复请求后碎片化严重直至 OOM。

解决方案:
  1. 所有输入张量在进入 GPU 前统一 padding/crop 到模型要求的 (C, H, W)
  2. 记录原始尺寸，推理完成后反裁剪还原
  3. padding 使用零填充，避免引入虚假信号
"""
from typing import Tuple, Optional, Dict
import numpy as np
import torch


class TensorStandardizer:
    def __init__(
        self,
        target_height: int = 256,
        target_width: int = 256,
        padding_mode: str = "constant",
        padding_value: float = 0.0,
    ):
        self.target_height = target_height
        self.target_width = target_width
        self.padding_mode = padding_mode
        self.padding_value = padding_value

    def standardize_tensor(
        self,
        tensor: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        shape = tensor.shape
        ndim = len(shape)

        if ndim == 5:
            return self._standardize_5d(tensor)
        elif ndim == 4:
            return self._standardize_4d(tensor)
        elif ndim == 3:
            return self._standardize_3d(tensor)
        else:
            meta = {"original_shape": shape, "padded": False}
            return tensor, meta

    def _standardize_5d(
        self, tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:
        B, T, C, H, W = tensor.shape
        meta = {
            "original_shape": (B, T, C, H, W),
            "original_h": H,
            "original_w": W,
            "padded": False,
        }

        if H == self.target_height and W == self.target_width:
            return tensor, meta

        meta["padded"] = True

        pad_h = max(self.target_height - H, 0)
        pad_w = max(self.target_width - W, 0)

        if pad_h > 0 or pad_w > 0:
            padded = torch.nn.functional.pad(
                tensor,
                (0, pad_w, 0, pad_h),
                mode=self.padding_mode,
                value=self.padding_value,
            )
        else:
            padded = tensor

        if H > self.target_height or W > self.target_width:
            padded = self._center_crop_5d(padded, self.target_height, self.target_width)

        meta["final_shape"] = tuple(padded.shape)
        return padded, meta

    def _standardize_4d(
        self, tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:
        B, C, H, W = tensor.shape
        meta = {
            "original_shape": (B, C, H, W),
            "original_h": H,
            "original_w": W,
            "padded": False,
        }

        if H == self.target_height and W == self.target_width:
            return tensor, meta

        meta["padded"] = True

        pad_h = max(self.target_height - H, 0)
        pad_w = max(self.target_width - W, 0)

        if pad_h > 0 or pad_w > 0:
            padded = torch.nn.functional.pad(
                tensor,
                (0, pad_w, 0, pad_h),
                mode=self.padding_mode,
                value=self.padding_value,
            )
        else:
            padded = tensor

        if H > self.target_height or W > self.target_width:
            padded = self._center_crop_4d(padded, self.target_height, self.target_width)

        meta["final_shape"] = tuple(padded.shape)
        return padded, meta

    def _standardize_3d(
        self, tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:
        C, H, W = tensor.shape
        meta = {
            "original_shape": (C, H, W),
            "original_h": H,
            "original_w": W,
            "padded": False,
        }

        if H == self.target_height and W == self.target_width:
            return tensor, meta

        meta["padded"] = True

        pad_h = max(self.target_height - H, 0)
        pad_w = max(self.target_width - W, 0)

        if pad_h > 0 or pad_w > 0:
            padded = torch.nn.functional.pad(
                tensor,
                (0, pad_w, 0, pad_h),
                mode=self.padding_mode,
                value=self.padding_value,
            )
        else:
            padded = tensor

        if H > self.target_height or W > self.target_width:
            padded = self._center_crop_3d(padded, self.target_height, self.target_width)

        meta["final_shape"] = tuple(padded.shape)
        return padded, meta

    def restore_tensor(
        self,
        tensor: torch.Tensor,
        meta: Dict,
    ) -> torch.Tensor:
        if not meta.get("padded", False):
            return tensor

        orig_h = meta["original_h"]
        orig_w = meta["original_w"]
        ndim = len(tensor.shape)

        if ndim == 5:
            return self._restore_5d(tensor, orig_h, orig_w)
        elif ndim == 4:
            return self._restore_4d(tensor, orig_h, orig_w)
        elif ndim == 3:
            return self._restore_3d(tensor, orig_h, orig_w)

        return tensor

    @staticmethod
    def _center_crop_5d(tensor: torch.Tensor, th: int, tw: int) -> torch.Tensor:
        _, _, _, H, W = tensor.shape
        h_off = (H - th) // 2
        w_off = (W - tw) // 2
        return tensor[:, :, :, h_off : h_off + th, w_off : w_off + tw]

    @staticmethod
    def _center_crop_4d(tensor: torch.Tensor, th: int, tw: int) -> torch.Tensor:
        _, _, H, W = tensor.shape
        h_off = (H - th) // 2
        w_off = (W - tw) // 2
        return tensor[:, :, h_off : h_off + th, w_off : w_off + tw]

    @staticmethod
    def _center_crop_3d(tensor: torch.Tensor, th: int, tw: int) -> torch.Tensor:
        _, H, W = tensor.shape
        h_off = (H - th) // 2
        w_off = (W - tw) // 2
        return tensor[:, h_off : h_off + th, w_off : w_off + tw]

    @staticmethod
    def _restore_5d(tensor: torch.Tensor, orig_h: int, orig_w: int) -> torch.Tensor:
        _, _, _, H, W = tensor.shape
        h_off = (H - orig_h) // 2
        w_off = (W - orig_w) // 2
        if h_off == 0 and w_off == 0 and H == orig_h and W == orig_w:
            return tensor
        h_start = max(h_off, 0)
        w_start = max(w_off, 0)
        h_end = min(h_start + orig_h, H)
        w_end = min(w_start + orig_w, W)
        return tensor[:, :, :, h_start:h_end, w_start:w_end]

    @staticmethod
    def _restore_4d(tensor: torch.Tensor, orig_h: int, orig_w: int) -> torch.Tensor:
        _, _, H, W = tensor.shape
        h_off = (H - orig_h) // 2
        w_off = (W - orig_w) // 2
        if h_off == 0 and w_off == 0 and H == orig_h and W == orig_w:
            return tensor
        h_start = max(h_off, 0)
        w_start = max(w_off, 0)
        h_end = min(h_start + orig_h, H)
        w_end = min(w_start + orig_w, W)
        return tensor[:, :, h_start:h_end, w_start:w_end]

    @staticmethod
    def _restore_3d(tensor: torch.Tensor, orig_h: int, orig_w: int) -> torch.Tensor:
        _, H, W = tensor.shape
        h_off = (H - orig_h) // 2
        w_off = (W - orig_w) // 2
        if h_off == 0 and w_off == 0 and H == orig_h and W == orig_w:
            return tensor
        h_start = max(h_off, 0)
        w_start = max(w_off, 0)
        h_end = min(h_start + orig_h, H)
        w_end = min(w_start + orig_w, W)
        return tensor[:, h_start:h_end, w_start:w_end]

    def standardize_numpy(
        self,
        data: np.ndarray,
    ) -> Tuple[np.ndarray, Dict]:
        tensor = torch.from_numpy(data).float() if isinstance(data, np.ndarray) else data.float()
        standardized, meta = self.standardize_tensor(tensor)
        return standardized.numpy(), meta
