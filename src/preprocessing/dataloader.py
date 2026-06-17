"""
雷达扫描序列时间对齐与采样
"""
import os
import glob
from datetime import datetime, timedelta
from typing import List, Optional, Tuple
import numpy as np

from ..radar_parser.parser import RadarBinaryParser
from ..radar_parser.coordinates import PolarToCartesian


class RadarDataLoader:
    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self.parser = RadarBinaryParser(self.config.get("radar", {}))
        self.converter = PolarToCartesian(
            grid_size=self.config.get("preprocessing", {}).get("grid_size", 256),
            max_range_km=self.config.get("radar", {}).get("max_range_km", 460),
        )
        self.time_interval_minutes = self.config.get("preprocessing", {}).get("time_interval_minutes", 5)
        self.variables = self.config.get("radar", {}).get("polar_variables", ["Z", "ZDR"])

    def load_from_directory(
        self,
        dir_path: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        pattern: str = "*.bin",
    ) -> List[dict]:
        files = sorted(glob.glob(os.path.join(dir_path, pattern)))
        return self._load_and_filter(files, start_time, end_time)

    def load_from_files(self, file_paths: List[str]) -> List[dict]:
        return self._load_and_filter(sorted(file_paths), None, None)

    def load_single_file(self, file_path: str) -> Optional[dict]:
        try:
            volume = self.parser.parse_file(file_path)
            return self._volume_to_frame(volume)
        except Exception:
            return None

    def load_from_bytes_list(self, bytes_list: List[Tuple[str, bytes]]) -> List[dict]:
        frames = []
        for filename, data in bytes_list:
            try:
                volume = self.parser.parse_bytes(data, filename)
                frame = self._volume_to_frame(volume)
                if frame is not None:
                    frames.append(frame)
            except Exception:
                continue
        frames.sort(key=lambda f: f["timestamp"])
        return frames

    def _load_and_filter(
        self,
        file_paths: List[str],
        start_time: Optional[datetime],
        end_time: Optional[datetime],
    ) -> List[dict]:
        frames = []
        for fp in file_paths:
            try:
                volume = self.parser.parse_file(fp)
                frame = self._volume_to_frame(volume)
                if frame is None:
                    continue
                if start_time and frame["timestamp"] < start_time:
                    continue
                if end_time and frame["timestamp"] > end_time:
                    continue
                frames.append(frame)
            except Exception:
                continue
        frames.sort(key=lambda f: f["timestamp"])
        return frames

    def _volume_to_frame(self, volume) -> Optional[dict]:
        if not volume.sweeps:
            return None

        var_data = {}
        for var_name in self.variables:
            cart = self.converter.convert_volume(volume, var_name=var_name, sweep_idx=0)
            if cart is not None:
                var_data[var_name] = cart.astype(np.float32)

        if not var_data:
            return None

        return {
            "timestamp": volume.scan_time,
            "radar_id": volume.radar_id,
            "data": var_data,
        }
