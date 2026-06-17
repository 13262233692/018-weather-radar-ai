"""
雷达基数据模型定义
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
from datetime import datetime


@dataclass
class RadialData:
    azimuth: float
    elevation: float
    time: datetime
    variables: Dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class SweepData:
    sweep_number: int
    elevation_angle: float
    radials: List[RadialData] = field(default_factory=list)


@dataclass
class RadarVolume:
    radar_id: str
    radar_name: str
    latitude: float
    longitude: float
    altitude: float
    scan_time: datetime
    sweeps: List[SweepData] = field(default_factory=list)

    def get_variable(self, var_name: str, sweep_idx: int = 0) -> Optional[np.ndarray]:
        if sweep_idx >= len(self.sweeps):
            return None
        sweep = self.sweeps[sweep_idx]
        if not sweep.radials:
            return None
        if var_name not in sweep.radials[0].variables:
            return None
        data = np.array([r.variables[var_name] for r in sweep.radials])
        return data

    def list_variables(self) -> List[str]:
        if not self.sweeps or not self.sweeps[0].radials:
            return []
        return list(self.sweeps[0].radials[0].variables.keys())
