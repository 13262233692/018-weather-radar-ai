"""
FY-4 卫星数据预处理模块

功能:
  1. 将卫星影像重采样到与雷达数据相同的 256×256 网格
  2. 与雷达数据进行地理坐标配准
  3. 按雷达扫描时间进行时间对齐插值
  4. 红外亮温转归一化特征
  5. 生成硬注意力掩码（Mask）张量
"""
from typing import Optional, List, Tuple
from datetime import datetime, timedelta
import numpy as np
import torch

from .fy4_parser import FY4IRData, FY4IRChannel


class SatellitePreprocessor:
    def __init__(
        self,
        target_height: int = 256,
        target_width: int = 256,
        radar_lat: float = None,
        radar_lon: float = None,
        radar_range_km: float = 460.0,
    ):
        self.target_height = target_height
        self.target_width = target_width
        self.radar_lat = radar_lat
        self.radar_lon = radar_lon
        self.radar_range_km = radar_range_km

        self._tbb_min = 180.0
        self._tbb_max = 320.0
        self._conv_temp_threshold = 220.0

    def preprocess_for_fusion(
        self,
        fy4_data: FY4IRData,
        target_size: Optional[Tuple[int, int]] = None,
    ) -> Optional[torch.Tensor]:
        if not fy4_data.channels:
            return None

        if target_size is None:
            target_size = (self.target_height, self.target_width)
        th, tw = target_size

        ir1 = fy4_data.get_brightness_temperature(FY4IRChannel.IR1)
        ir2 = fy4_data.get_brightness_temperature(FY4IRChannel.IR2)

        if ir1.size == 0:
            return None

        ir1_resized = self._resize_to_target(ir1, th, tw)
        ir2_resized = self._resize_to_target(ir2, th, tw) if ir2.size > 0 else np.zeros_like(ir1_resized)

        ir1_norm = self._normalize_tbb(ir1_resized)
        ir2_norm = self._normalize_tbb(ir2_resized)

        features = np.stack([ir1_norm, ir2_norm], axis=0)
        return torch.from_numpy(features).float()

    def generate_attention_mask(
        self,
        fy4_data: FY4IRData,
        target_size: Optional[Tuple[int, int]] = None,
        method: str = "hard",
    ) -> Optional[torch.Tensor]:
        if not fy4_data.channels:
            return None

        if target_size is None:
            target_size = (self.target_height, self.target_width)
        th, tw = target_size

        ir1 = fy4_data.get_brightness_temperature(FY4IRChannel.IR1)
        if ir1.size == 0:
            return None

        ir1_resized = self._resize_to_target(ir1, th, tw)

        if method == "hard":
            mask = self._generate_hard_attention_mask(ir1_resized)
        elif method == "soft":
            mask = self._generate_soft_attention_mask(ir1_resized)
        else:
            mask = self._generate_threshold_mask(ir1_resized)

        return torch.from_numpy(mask).float().unsqueeze(0)

    def _generate_hard_attention_mask(self, tbb: np.ndarray) -> np.ndarray:
        temp_diff = self._conv_temp_threshold - tbb
        convective_mask = (temp_diff > 0).astype(np.float32)

        gradient_y, gradient_x = np.gradient(tbb)
        gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
        edge_mask = (gradient_magnitude > 0.5).astype(np.float32)

        combined = np.maximum(convective_mask, edge_mask * 0.5)
        combined = np.clip(combined, 0.0, 1.0)
        combined = self._smooth_mask(combined, kernel_size=3)

        combined = 0.5 + 0.5 * combined
        combined = np.clip(combined, 0.1, 2.0)

        return combined.astype(np.float32)

    def _generate_soft_attention_mask(self, tbb: np.ndarray) -> np.ndarray:
        temp_diff = self._conv_temp_threshold - tbb
        temp_diff = np.clip(temp_diff, 0, None)

        attention = 1.0 + (temp_diff / 100.0) ** 0.5
        attention = np.clip(attention, 0.5, 3.0)

        return attention.astype(np.float32)

    def _generate_threshold_mask(self, tbb: np.ndarray) -> np.ndarray:
        mask = np.ones_like(tbb, dtype=np.float32)
        cold_mask = tbb < 210
        mask[cold_mask] = 2.0
        warm_mask = tbb > 260
        mask[warm_mask] = 0.5
        return mask

    def preprocess_sequence(
        self,
        fy4_sequence: List[FY4IRData],
        radar_timestamps: List[datetime],
        target_size: Optional[Tuple[int, int]] = None,
    ) -> Optional[torch.Tensor]:
        if target_size is None:
            target_size = (self.target_height, self.target_width)

        if not fy4_sequence:
            return None

        satellite_times = [d.metadata.start_time for d in fy4_sequence]

        features_list = []
        for radar_time in radar_timestamps:
            matched_data = self._match_by_time(fy4_sequence, satellite_times, radar_time)
            if matched_data is None:
                features = torch.zeros((2,) + target_size, dtype=torch.float32)
            else:
                features = self.preprocess_for_fusion(matched_data, target_size)
                if features is None:
                    features = torch.zeros((2,) + target_size, dtype=torch.float32)
            features_list.append(features)

        return torch.stack(features_list, dim=0).unsqueeze(0)

    def generate_mask_sequence(
        self,
        fy4_sequence: List[FY4IRData],
        radar_timestamps: List[datetime],
        target_size: Optional[Tuple[int, int]] = None,
        method: str = "hard",
    ) -> Optional[torch.Tensor]:
        if target_size is None:
            target_size = (self.target_height, self.target_width)

        if not fy4_sequence:
            return None

        satellite_times = [d.metadata.start_time for d in fy4_sequence]

        masks_list = []
        for radar_time in radar_timestamps:
            matched_data = self._match_by_time(fy4_sequence, satellite_times, radar_time)
            if matched_data is None:
                mask = torch.ones((1,) + target_size, dtype=torch.float32)
            else:
                mask = self.generate_attention_mask(matched_data, target_size, method)
                if mask is None:
                    mask = torch.ones((1,) + target_size, dtype=torch.float32)
            masks_list.append(mask)

        return torch.stack(masks_list, dim=0).unsqueeze(0)

    def _match_by_time(
        self,
        fy4_sequence: List[FY4IRData],
        satellite_times: List[datetime],
        target_time: datetime,
    ) -> Optional[FY4IRData]:
        if not fy4_sequence:
            return None

        time_diffs = [abs((t - target_time).total_seconds()) for t in satellite_times]
        min_idx = int(np.argmin(time_diffs))

        min_diff = time_diffs[min_idx]
        if min_diff > 3600:
            return None

        return fy4_sequence[min_idx]

    def _resize_to_target(
        self,
        data: np.ndarray,
        th: int,
        tw: int,
    ) -> np.ndarray:
        h, w = data.shape

        if h == th and w == tw:
            return data

        y_idx = np.linspace(0, h - 1, th).astype(np.int32)
        x_idx = np.linspace(0, w - 1, tw).astype(np.int32)

        y_idx = np.clip(y_idx, 0, h - 1)
        x_idx = np.clip(x_idx, 0, w - 1)

        resized = data[y_idx[:, None], x_idx[None, :]]
        return resized

    def _normalize_tbb(self, tbb: np.ndarray) -> np.ndarray:
        normalized = (tbb - self._tbb_min) / (self._tbb_max - self._tbb_min)
        normalized = np.clip(normalized, 0.0, 1.0)
        return normalized.astype(np.float32)

    def _denormalize_tbb(self, normalized: np.ndarray) -> np.ndarray:
        return normalized * (self._tbb_max - self._tbb_min) + self._tbb_min

    @staticmethod
    def _smooth_mask(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        if kernel_size <= 1:
            return mask

        from scipy.ndimage import uniform_filter

        try:
            smoothed = uniform_filter(mask, size=kernel_size, mode="nearest")
            return smoothed
        except Exception:
            return mask

    @staticmethod
    def align_coordinates(
        satellite_data: np.ndarray,
        radar_data: np.ndarray,
        radar_lat: float,
        radar_lon: float,
        radar_range_km: float = 460.0,
    ) -> np.ndarray:
        sh, sw = satellite_data.shape
        rh, rw = radar_data.shape

        lat_km_per_deg = 111.0
        lon_km_per_deg = 111.0 * np.cos(np.radians(radar_lat))

        half_lat = radar_range_km / lat_km_per_deg / 2.0
        half_lon = radar_range_km / lon_km_per_deg / 2.0

        y, x = np.ogrid[:sh, :sw]
        lat = 60.0 - y * (120.0 / sh)
        lon = x * (180.0 / sw)

        mask_lat = (lat > radar_lat - half_lat) & (lat < radar_lat + half_lat)
        mask_lon = (lon > radar_lon - half_lon) & (lon < radar_lon + half_lon)
        mask = mask_lat & mask_lon

        aligned = np.full((rh, rw), 280.0, dtype=np.float32)

        try:
            sat_y = np.where(mask)[0]
            sat_x = np.where(mask)[1]

            if len(sat_y) > 0 and len(sat_x) > 0:
                sat_region = satellite_data[sat_y.min():sat_y.max()+1, sat_x.min():sat_x.max()+1]

                from scipy.ndimage import zoom

                zoom_h = rh / sat_region.shape[0]
                zoom_w = rw / sat_region.shape[1]

                if zoom_h > 0 and zoom_w > 0:
                    resized = zoom(sat_region, (zoom_h, zoom_w), order=1)
                    if resized.shape == (rh, rw):
                        aligned = resized
        except Exception:
            pass

        return aligned
