"""
张量序列构建 - 将雷达扫描帧转化为模型输入张量
"""
from typing import List, Optional, Dict
import numpy as np
import torch

from .normalizer import RadarDataNormalizer


class TensorSequenceBuilder:
    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self.normalizer = RadarDataNormalizer(self.config.get("preprocessing", {}))

        model_cfg = self.config.get("model", {})
        self.input_seq_len = model_cfg.get("input_seq_len", 24)
        self.output_seq_len = model_cfg.get("output_seq_len", 12)
        self.input_channels = model_cfg.get("input_channels", 2)
        self.img_height = model_cfg.get("img_height", 256)
        self.img_width = model_cfg.get("img_width", 256)
        self.channels = self.config.get("radar", {}).get("polar_variables", ["Z", "ZDR"])[: self.input_channels]

    def build_input_tensor(self, frames: List[dict]) -> Optional[torch.Tensor]:
        if len(frames) < self.input_seq_len:
            frames = self._pad_frames(frames, self.input_seq_len)

        selected_frames = frames[-self.input_seq_len :]

        sequence = []
        for frame in selected_frames:
            channel_data = self.normalizer.normalize_channel(frame["data"], self.channels)
            sequence.append(channel_data)

        tensor = np.stack(sequence, axis=0)
        tensor = torch.from_numpy(tensor).float()
        tensor = tensor.unsqueeze(0)

        return tensor

    def build_training_pair(
        self, frames: List[dict]
    ) -> Optional[Dict[str, torch.Tensor]]:
        required_len = self.input_seq_len + self.output_seq_len
        if len(frames) < required_len:
            return None

        input_frames = frames[: self.input_seq_len]
        target_frames = frames[self.input_seq_len : required_len]

        input_tensor = self.build_input_tensor(input_frames)
        if input_tensor is None:
            return None

        target_sequence = []
        for frame in target_frames:
            channel_data = self.normalizer.normalize_channel(frame["data"], self.channels)
            target_sequence.append(channel_data)

        target_tensor = np.stack(target_sequence, axis=0)
        target_tensor = torch.from_numpy(target_tensor).float()
        target_tensor = target_tensor.unsqueeze(0)

        return {"inputs": input_tensor, "targets": target_tensor}

    def denormalize_output(self, output_tensor: torch.Tensor, var_name: str = "Z") -> np.ndarray:
        output_np = output_tensor.squeeze().cpu().numpy()
        if output_np.ndim == 4:
            output_np = output_np[:, 0]

        denormalized = []
        for frame in output_np:
            denorm = self.normalizer.denormalize(frame, var_name)
            denormalized.append(denorm)

        return np.stack(denormalized, axis=0)

    def _pad_frames(self, frames: List[dict], target_len: int) -> List[dict]:
        if not frames:
            raise ValueError("Cannot pad empty frames")

        padded = list(frames)
        while len(padded) < target_len:
            padded.insert(0, padded[0])
        return padded
