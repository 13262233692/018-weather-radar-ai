"""
核心服务依赖 - 模型、解析器、渲染器的单例管理
"""
import os
from typing import Optional

from ..utils.config import load_config
from ..radar_parser.parser import RadarBinaryParser
from ..radar_parser.coordinates import PolarToCartesian
from ..preprocessing.normalizer import RadarDataNormalizer
from ..preprocessing.dataloader import RadarDataLoader
from ..preprocessing.tensor_builder import TensorSequenceBuilder
from ..models.convlstm_model import WeatherRadarPredictor
from ..visualization.renderer import RadarImageRenderer


class AppState:
    def __init__(self, config_path: str = None):
        self.config = load_config(config_path)
        self.parser: Optional[RadarBinaryParser] = None
        self.converter: Optional[PolarToCartesian] = None
        self.normalizer: Optional[RadarDataNormalizer] = None
        self.dataloader: Optional[RadarDataLoader] = None
        self.tensor_builder: Optional[TensorSequenceBuilder] = None
        self.predictor: Optional[WeatherRadarPredictor] = None
        self.renderer: Optional[RadarImageRenderer] = None
        self._initialized = False

    def initialize(self):
        if self._initialized:
            return

        self.parser = RadarBinaryParser(self.config.get("radar", {}))
        self.converter = PolarToCartesian(
            grid_size=self.config.get("preprocessing", {}).get("grid_size", 256),
            max_range_km=self.config.get("radar", {}).get("max_range_km", 460),
        )
        self.normalizer = RadarDataNormalizer(self.config.get("preprocessing", {}))
        self.dataloader = RadarDataLoader(self.config)
        self.tensor_builder = TensorSequenceBuilder(self.config)
        self.predictor = WeatherRadarPredictor(self.config)
        self.renderer = RadarImageRenderer(self.config)

        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        os.makedirs(os.path.join(base_dir, "data"), exist_ok=True)
        os.makedirs(os.path.join(base_dir, "checkpoints"), exist_ok=True)

        self._initialized = True

    @property
    def is_model_loaded(self) -> bool:
        return self.predictor is not None and self._initialized

    @property
    def device(self) -> str:
        if self.predictor and hasattr(self.predictor, "device"):
            return str(self.predictor.device)
        return "cpu"


_app_state: Optional[AppState] = None


def get_app_state() -> AppState:
    global _app_state
    if _app_state is None:
        _app_state = AppState()
        _app_state.initialize()
    return _app_state
