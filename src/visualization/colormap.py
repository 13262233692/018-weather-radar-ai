"""
气象专用色表定义 - 用于雷达回波伪彩色渲染
"""
import numpy as np
from typing import List, Tuple


def create_weather_colormap() -> List[Tuple[float, Tuple[float, float, float]]]:
    colors = [
        (-30.0, (0.0, 0.0, 0.0)),
        (-20.0, (0.0, 0.0, 0.4)),
        (-10.0, (0.0, 0.2, 0.6)),
        (0.0, (0.0, 0.4, 0.8)),
        (5.0, (0.0, 0.6, 0.4)),
        (10.0, (0.0, 0.8, 0.0)),
        (15.0, (0.4, 0.9, 0.0)),
        (20.0, (0.8, 1.0, 0.0)),
        (25.0, (1.0, 0.9, 0.0)),
        (30.0, (1.0, 0.7, 0.0)),
        (35.0, (1.0, 0.5, 0.0)),
        (40.0, (1.0, 0.3, 0.0)),
        (45.0, (1.0, 0.0, 0.0)),
        (50.0, (0.8, 0.0, 0.2)),
        (55.0, (0.7, 0.0, 0.5)),
        (60.0, (0.6, 0.0, 0.7)),
        (65.0, (0.5, 0.0, 0.8)),
        (70.0, (0.4, 0.0, 0.9)),
        (75.0, (1.0, 1.0, 1.0)),
    ]
    return colors


class WeatherColorMap:
    def __init__(self):
        self.color_stops = create_weather_colormap()
        self.values = np.array([c[0] for c in self.color_stops])
        self.colors = np.array([list(c[1]) for c in self.color_stops])

    def __call__(self, data: np.ndarray) -> np.ndarray:
        normalized = np.clip(data, self.values.min(), self.values.max())

        indices = np.searchsorted(self.values, normalized) - 1
        indices = np.clip(indices, 0, len(self.values) - 2)

        v0 = self.values[indices]
        v1 = self.values[indices + 1]
        c0 = self.colors[indices]
        c1 = self.colors[indices + 1]

        t = np.where(v1 - v0 > 0, (normalized - v0) / (v1 - v0), 0.0)
        t = np.expand_dims(t, axis=-1)

        rgb = c0 + t * (c1 - c0)
        return (rgb * 255).astype(np.uint8)


def get_colorbar_labels() -> List[Tuple[float, str]]:
    return [
        (-30, "-30"),
        (-20, "-20"),
        (-10, "-10"),
        (0, "0"),
        (10, "10"),
        (20, "20"),
        (30, "30"),
        (40, "40"),
        (50, "50"),
        (60, "60"),
        (70, "70"),
        (75, "75 dBZ"),
    ]
