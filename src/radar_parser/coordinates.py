"""
极坐标到笛卡尔坐标转换
"""
import numpy as np
from typing import Optional

from .models import RadarVolume


class PolarToCartesian:
    def __init__(self, grid_size: int = 256, max_range_km: float = 460.0):
        self.grid_size = grid_size
        self.max_range_km = max_range_km
        self._grid_x, self._grid_y = self._build_grid()

    def _build_grid(self):
        half = self.grid_size // 2
        x = np.linspace(-self.max_range_km, self.max_range_km, self.grid_size)
        y = np.linspace(-self.max_range_km, self.max_range_km, self.grid_size)
        xx, yy = np.meshgrid(x, y)
        return xx, yy

    def convert_volume(self, volume: RadarVolume, var_name: str = "Z", sweep_idx: int = 0) -> Optional[np.ndarray]:
        polar_data = volume.get_variable(var_name, sweep_idx)
        if polar_data is None:
            return None

        if not volume.sweeps or not volume.sweeps[sweep_idx].radials:
            return None

        radials = volume.sweeps[sweep_idx].radials
        azimuths = np.array([r.azimuth for r in radials])
        gate_count = polar_data.shape[1]
        gate_km = self.max_range_km / gate_count
        ranges = np.arange(gate_count) * gate_km + gate_km / 2

        return self._interpolate_polar_to_cartesian(polar_data, azimuths, ranges)

    def convert_polar_data(
        self,
        polar_data: np.ndarray,
        azimuths: np.ndarray,
        ranges: np.ndarray,
    ) -> np.ndarray:
        return self._interpolate_polar_to_cartesian(polar_data, azimuths, ranges)

    def _interpolate_polar_to_cartesian(
        self,
        polar_data: np.ndarray,
        azimuths: np.ndarray,
        ranges: np.ndarray,
    ) -> np.ndarray:
        grid_r = np.sqrt(self._grid_x ** 2 + self._grid_y ** 2)
        grid_theta = np.degrees(np.arctan2(self._grid_x, self._grid_y)) % 360.0

        valid_mask = grid_r <= self.max_range_km

        cart_data = np.full((self.grid_size, self.grid_size), np.nan, dtype=np.float32)

        if len(azimuths) < 2 or len(ranges) < 2:
            return cart_data

        azimuth_idx = np.searchsorted(np.sort(azimuths), grid_theta[valid_mask])
        azimuth_idx = np.clip(azimuth_idx, 0, len(azimuths) - 1)

        range_idx = np.searchsorted(ranges, grid_r[valid_mask])
        range_idx = np.clip(range_idx, 0, len(ranges) - 1)

        azimuth_order = np.argsort(azimuths)
        sorted_polar = polar_data[azimuth_order]

        cart_data[valid_mask] = sorted_polar[azimuth_idx, range_idx]

        cart_data = self._fill_missing(cart_data)

        return cart_data

    @staticmethod
    def _fill_missing(data: np.ndarray) -> np.ndarray:
        from scipy.ndimage import map_coordinates, distance_transform_edt

        mask = np.isnan(data)
        if not mask.any():
            return data

        try:
            indices = distance_transform_edt(mask, return_indices=True)[1]
            filled = data.copy()
            filled[mask] = data[indices[0][mask], indices[1][mask]]
            return filled
        except Exception:
            filled = data.copy()
            filled[np.isnan(filled)] = 0.0
            return filled
