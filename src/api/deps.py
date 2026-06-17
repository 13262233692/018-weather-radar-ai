"""
核心服务依赖 - 模型、解析器、渲染器、推理调度器的单例管理

重构要点:
  1. 将 WeatherRadarPredictor 替换为 DynamicBatchScheduler
  2. 推理不再直接调用 model.forward(), 而是通过 submit() 提交到批处理队列
  3. 所有推理请求经过 TensorStandardizer 标准化后统一尺寸
  4. GPUMemoryManager 负责推理前后的显存管理
"""
import os
import asyncio
from typing import Optional

import torch

from ..utils.config import load_config
from ..radar_parser.parser import RadarBinaryParser
from ..radar_parser.coordinates import PolarToCartesian
from ..preprocessing.normalizer import RadarDataNormalizer
from ..preprocessing.dataloader import RadarDataLoader
from ..preprocessing.tensor_builder import TensorSequenceBuilder
from ..preprocessing.tensor_standardizer import TensorStandardizer
from ..models.convlstm_model import ConvLSTMModel
from ..visualization.renderer import RadarImageRenderer
from ..inference.gpu_manager import GPUMemoryManager
from ..inference.batch_scheduler import DynamicBatchScheduler


class AppState:
    def __init__(self, config_path: str = None):
        self.config = load_config(config_path)
        self.parser: Optional[RadarBinaryParser] = None
        self.converter: Optional[PolarToCartesian] = None
        self.normalizer: Optional[RadarDataNormalizer] = None
        self.dataloader: Optional[RadarDataLoader] = None
        self.tensor_builder: Optional[TensorSequenceBuilder] = None
        self.standardizer: Optional[TensorStandardizer] = None
        self.model: Optional[ConvLSTMModel] = None
        self.gpu_manager: Optional[GPUMemoryManager] = None
        self.batch_scheduler: Optional[DynamicBatchScheduler] = None
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
        self.standardizer = TensorStandardizer(
            target_height=self.config.get("model", {}).get("img_height", 256),
            target_width=self.config.get("model", {}).get("img_width", 256),
        )

        model_cfg = self.config.get("model", {})
        gpu_cfg = self.config.get("gpu", {})

        if gpu_cfg.get("device", "auto") == "auto":
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device_str = gpu_cfg.get("device", "cpu")
        self.device = torch.device(device_str)

        self.model = ConvLSTMModel(
            input_channels=model_cfg.get("input_channels", 2),
            hidden_channels=model_cfg.get("hidden_channels", [64, 64, 64]),
            kernel_size=model_cfg.get("kernel_size", 3),
            num_layers=model_cfg.get("num_layers", 3),
            input_seq_len=model_cfg.get("input_seq_len", 24),
            output_seq_len=model_cfg.get("output_seq_len", 12),
            img_height=model_cfg.get("img_height", 256),
            img_width=model_cfg.get("img_width", 256),
        ).to(self.device)

        checkpoint_path = model_cfg.get("checkpoint_path", "./checkpoints/convlstm_weather.pth")
        if os.path.exists(checkpoint_path):
            try:
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                if "model_state_dict" in checkpoint:
                    self.model.load_state_dict(checkpoint["model_state_dict"])
                else:
                    self.model.load_state_dict(checkpoint)
            except Exception:
                pass
        self.model.eval()

        self.gpu_manager = GPUMemoryManager(self.config)

        self.batch_scheduler = DynamicBatchScheduler(
            model=self.model,
            device=self.device,
            config=self.config,
        )

        self.renderer = RadarImageRenderer(self.config)

        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        os.makedirs(os.path.join(base_dir, "data"), exist_ok=True)
        os.makedirs(os.path.join(base_dir, "checkpoints"), exist_ok=True)

        self._initialized = True

    async def start_scheduler(self):
        if self.batch_scheduler is not None:
            await self.batch_scheduler.start()

    async def stop_scheduler(self):
        if self.batch_scheduler is not None:
            await self.batch_scheduler.stop()

    @property
    def is_model_loaded(self) -> bool:
        return self.model is not None and self._initialized

    @property
    def device_str(self) -> str:
        return str(self.device) if hasattr(self, "device") else "cpu"


_app_state: Optional[AppState] = None


def get_app_state() -> AppState:
    global _app_state
    if _app_state is None:
        _app_state = AppState()
        _app_state.initialize()
    return _app_state
