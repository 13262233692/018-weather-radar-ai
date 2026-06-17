"""
雷达数据归一化处理
"""
import numpy as np
from typing import Dict, Optional


class RadarDataNormalizer:
    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self.normalization_params = self.config.get("normalization", {
            "Z": {"min": -30, "max": 75},
            "ZDR": {"min": -2, "max": 8},
        })

    def normalize(self, data: np.ndarray, var_name: str = "Z") -> np.ndarray:
        params = self.normalization_params.get(var_name, {"min": -30, "max": 75})
        vmin, vmax = params["min"], params["max"]

        normalized = (data - vmin) / (vmax - vmin)
        normalized = np.clip(normalized, 0.0, 1.0)
        normalized = normalized.astype(np.float32)

        return normalized

    def denormalize(self, normalized: np.ndarray, var_name: str = "Z") -> np.ndarray:
        params = self.normalization_params.get(var_name, {"min": -30, "max": 75})
        vmin, vmax = params["min"], params["max"]

        data = normalized * (vmax - vmin) + vmin
        return data.astype(np.float32)

    def normalize_channel(self, data: Dict[str, np.ndarray], channels: list = None) -> np.ndarray:
        if channels is None:
            channels = list(data.keys())

        normalized_channels = []
        for ch in channels:
            if ch in data:
                normalized_channels.append(self.normalize(data[ch], ch))
            else:
                h, w = list(data.values())[0].shape
                normalized_channels.append(np.zeros((h, w), dtype=np.float32))

        return np.stack(normalized_channels, axis=0)
