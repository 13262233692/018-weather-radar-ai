"""
核心服务依赖 - 模型、解析器、渲染器、推理调度器的单例管理

重构要点:
  1. 使用 MultimodalConvLSTM 替换原有 ConvLSTMModel
  2. 新增 FY-4 卫星解析器和预处理器
  3. 所有推理请求通过 batch_scheduler.submit(radar, satellite, mask) 异步提交
"""
import os
from typing import Optional

import torch

from ..utils.config import load_config
from ..radar_parser.parser import RadarBinaryParser
from ..radar_parser.coordinates import PolarToCartesian
from ..preprocessing.normalizer import RadarDataNormalizer
from ..preprocessing.dataloader import RadarDataLoader
from ..preprocessing.tensor_builder import TensorSequenceBuilder
from ..preprocessing.tensor_standardizer import TensorStandardizer
from ..satellite.fy4_parser import FY4DataParser
from ..satellite.preprocessor import SatellitePreprocessor
from ..models.multimodal_fusion import MultimodalConvLSTM, FusionMode
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
        self.fy4_parser: Optional[FY4DataParser] = None
        self.fy4_preprocessor: Optional[SatellitePreprocessor] = None
        self.model: Optional[MultimodalConvLSTM] = None
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

        self.fy4_parser = FY4DataParser(self.config)
        self.fy4_preprocessor = SatellitePreprocessor(
            target_height=self.config.get("model", {}).get("img_height", 256),
            target_width=self.config.get("model", {}).get("img_width", 256),
            radar_range_km=self.config.get("radar", {}).get("max_range_km", 460),
        )

        model_cfg = self.config.get("model", {})
        fusion_cfg = self.config.get("fusion", {})
        gpu_cfg = self.config.get("gpu", {})

        if gpu_cfg.get("device", "auto") == "auto":
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device_str = gpu_cfg.get("device", "cpu")
        self.device = torch.device(device_str)

        self.use_satellite = fusion_cfg.get("enabled", True)
        self.fusion_mode = fusion_cfg.get("mode", FusionMode.HARD_MASK)

        self.model = MultimodalConvLSTM(
            radar_channels=model_cfg.get("input_channels", 2),
            satellite_channels=fusion_cfg.get("satellite_channels", 2),
            hidden_channels=model_cfg.get("hidden_channels", [64, 64, 64]),
            kernel_size=model_cfg.get("kernel_size", 3),
            num_layers=model_cfg.get("num_layers", 3),
            input_seq_len=model_cfg.get("input_seq_len", 24),
            output_seq_len=model_cfg.get("output_seq_len", 12),
            img_height=model_cfg.get("img_height", 256),
            img_width=model_cfg.get("img_width", 256),
            fusion_mode=self.fusion_mode,
            use_satellite=self.use_satellite,
        ).to(self.device)

        self._load_checkpoint()
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

    def _load_checkpoint(self):
        fusion_cfg = self.config.get("fusion", {})
        model_cfg = self.config.get("model", {})

        multimodal_checkpoint = fusion_cfg.get(
            "checkpoint_path", "./checkpoints/multimodal_convlstm.pth"
        )
        if os.path.exists(multimodal_checkpoint):
            try:
                checkpoint = torch.load(multimodal_checkpoint, map_location=self.device)
                if "model_state_dict" in checkpoint:
                    self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
                else:
                    self.model.load_state_dict(checkpoint, strict=False)
                return
            except Exception:
                pass

        radar_checkpoint = model_cfg.get(
            "checkpoint_path", "./checkpoints/convlstm_weather.pth"
        )
        if os.path.exists(radar_checkpoint):
            try:
                checkpoint = torch.load(radar_checkpoint, map_location=self.device)
                if "model_state_dict" in checkpoint:
                    state_dict = checkpoint["model_state_dict"]
                else:
                    state_dict = checkpoint

                encoder_keys = {}
                for k, v in state_dict.items():
                    if "encoder_layers" in k or "decoder_layers" in k or "output_conv" in k:
                        encoder_keys[k] = v

                self.model.load_state_dict(encoder_keys, strict=False)
            except Exception:
                pass

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
