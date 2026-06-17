"""
核心业务服务编排 - 串联解析、预处理、预测、可视化全流程

重构要点:
  1. 推理调用改为通过 BatchScheduler 异步提交
  2. 所有输入张量经 TensorStandardizer 标准化
  3. 命令行模式直接使用 GPUMemoryManager + model 推理
"""
import os
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Tuple
import numpy as np
import torch

from .utils.config import load_config, ensure_dir
from .radar_parser.parser import RadarBinaryParser
from .radar_parser.coordinates import PolarToCartesian
from .preprocessing.normalizer import RadarDataNormalizer
from .preprocessing.dataloader import RadarDataLoader
from .preprocessing.tensor_builder import TensorSequenceBuilder
from .preprocessing.tensor_standardizer import TensorStandardizer
from .satellite.fy4_parser import FY4DataParser
from .satellite.preprocessor import SatellitePreprocessor
from .models.multimodal_fusion import MultimodalConvLSTM, FusionMode
from .inference.gpu_manager import GPUMemoryManager
from .visualization.renderer import RadarImageRenderer


class WeatherRadarService:
    def __init__(self, config_path: str = None):
        self.config = load_config(config_path)
        self._init_components()

    def _init_components(self):
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
        self.gpu_manager = GPUMemoryManager(self.config)
        self.renderer = RadarImageRenderer(self.config)

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

    def _safe_predict(
        self,
        input_tensor: torch.Tensor,
        satellite_tensor: Optional[torch.Tensor] = None,
        hard_mask_tensor: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        standardized, meta = self.standardizer.standardize_tensor(input_tensor)

        total_channels = standardized.size(2)
        if satellite_tensor is not None:
            total_channels += satellite_tensor.size(2)

        estimated_mb = self.gpu_manager.estimate_batch_memory_mb(
            batch_size=standardized.size(0),
            seq_len=standardized.size(1),
            channels=total_channels,
            height=standardized.size(3),
            width=standardized.size(4),
            num_layers=len(self.model.encoder_layers),
        )

        if not self.gpu_manager.can_allocate(estimated_mb):
            self.gpu_manager.post_inference_cleanup()
            if not self.gpu_manager.can_allocate(estimated_mb):
                raise RuntimeError(
                    f"GPU OOM: cannot allocate {estimated_mb:.0f}MB. "
                    f"Free: {self.gpu_manager.get_stats().free_mb:.0f}MB"
                )

        sat_standardized = None
        mask_standardized = None
        sat_meta = None
        mask_meta = None

        if satellite_tensor is not None:
            sat_standardized, sat_meta = self.standardizer.standardize_tensor(satellite_tensor)
            sat_standardized = sat_standardized.to(self.device)

        if hard_mask_tensor is not None:
            mask_standardized, mask_meta = self.standardizer.standardize_tensor(hard_mask_tensor)
            mask_standardized = mask_standardized.to(self.device)

        with self.gpu_manager.safe_inference_context():
            standardized = standardized.to(self.device)
            with torch.no_grad():
                self.model.eval()
                output = self.model(
                    standardized,
                    satellite_tensor=sat_standardized,
                    hard_mask_tensor=mask_standardized,
                )
            output = output.cpu()

        restored = self.standardizer.restore_tensor(output, meta)
        return restored

    def process_files(
        self,
        file_paths: List[str],
        satellite_file_paths: Optional[List[str]] = None,
        save_images: bool = False,
        output_dir: str = None,
    ) -> Dict:
        frames = self.dataloader.load_from_files(file_paths)
        if not frames:
            return {"success": False, "error": "Failed to load any radar frames"}

        satellite_frames = None
        if satellite_file_paths and self.use_satellite:
            satellite_frames = []
            for f in satellite_file_paths:
                try:
                    data = self.fy4_parser.parse_file(f)
                    satellite_frames.append(data)
                except Exception:
                    continue
            if not satellite_frames:
                satellite_frames = None

        return self._process_frames(frames, satellite_frames, save_images, output_dir)

    def process_bytes(
        self,
        bytes_list: List[Tuple[str, bytes]],
        satellite_bytes_list: Optional[List[Tuple[str, bytes]]] = None,
        save_images: bool = False,
        output_dir: str = None,
    ) -> Dict:
        frames = self.dataloader.load_from_bytes_list(bytes_list)
        if not frames:
            return {"success": False, "error": "Failed to load any radar frames"}

        satellite_frames = None
        if satellite_bytes_list and self.use_satellite:
            satellite_frames = []
            for filename, content in satellite_bytes_list:
                try:
                    data = self.fy4_parser.parse_bytes(content, filename)
                    satellite_frames.append(data)
                except Exception:
                    continue
            if not satellite_frames:
                satellite_frames = None

        return self._process_frames(frames, satellite_frames, save_images, output_dir)

    def _process_frames(
        self,
        frames: List[dict],
        satellite_frames: Optional[List[dict]],
        save_images: bool,
        output_dir: Optional[str],
    ) -> Dict:
        result = {
            "success": True,
            "input_frame_count": len(frames),
            "radar_id": frames[0]["radar_id"],
            "start_time": frames[0]["timestamp"],
            "end_time": frames[-1]["timestamp"],
            "used_satellite_fusion": False,
        }

        input_tensor = self.tensor_builder.build_input_tensor(frames)
        if input_tensor is None:
            result["success"] = False
            result["error"] = (
                f"Insufficient frames. Need at least "
                f"{self.tensor_builder.input_seq_len}, got {len(frames)}"
            )
            return result

        satellite_tensor = None
        hard_mask_tensor = None

        if satellite_frames and self.use_satellite:
            preprocessed = self.fy4_preprocessor.preprocess_for_fusion(
                satellite_frames,
                target_seq_len=self.tensor_builder.input_seq_len,
                reference_times=[f["timestamp"] for f in frames],
            )
            satellite_tensor = preprocessed.get("satellite_tensor")
            hard_mask_tensor = preprocessed.get("hard_mask_tensor")
            if satellite_tensor is not None:
                result["used_satellite_fusion"] = True
                result["satellite_frame_count"] = len(satellite_frames)

        try:
            output_tensor = self._safe_predict(input_tensor, satellite_tensor, hard_mask_tensor)
        except RuntimeError as e:
            result["success"] = False
            result["error"] = str(e)
            return result

        denormalized = self.tensor_builder.denormalize_output(output_tensor, var_name="Z")

        interval = self.config.get("prediction", {}).get("interval_minutes", 10)
        pred_start = frames[-1]["timestamp"] + timedelta(minutes=interval)

        result["output_frame_count"] = len(denormalized)
        result["prediction_start"] = pred_start
        result["prediction_interval_minutes"] = interval
        result["prediction_timestamps"] = [
            pred_start + timedelta(minutes=i * interval)
            for i in range(len(denormalized))
        ]
        result["prediction_data"] = denormalized

        if save_images:
            if output_dir is None:
                base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                output_dir = os.path.join(base_dir, "output")
            ensure_dir(output_dir)

            image_paths = self._save_prediction_images(
                denormalized, pred_start, interval, frames[0]["radar_id"], output_dir
            )
            result["output_images"] = image_paths

        return result

    def _save_prediction_images(
        self,
        prediction_data: np.ndarray,
        start_time: datetime,
        interval: int,
        radar_id: str,
        output_dir: str,
    ) -> List[str]:
        image_paths = []
        images = self.renderer.render_sequence(
            prediction_data,
            start_time=start_time,
            interval_minutes=interval,
            radar_id=radar_id,
        )

        for i, img_bytes in enumerate(images):
            ts = start_time + timedelta(minutes=i * interval)
            filename = f"pred_{ts.strftime('%Y%m%d_%H%M')}_T{i * interval:03d}min.png"
            filepath = os.path.join(output_dir, filename)
            with open(filepath, "wb") as f:
                f.write(img_bytes)
            image_paths.append(filepath)

        gif_bytes = self.renderer.render_gif(
            prediction_data,
            start_time=start_time,
            interval_minutes=interval,
        )
        gif_path = os.path.join(output_dir, f"prediction_{start_time.strftime('%Y%m%d_%H%M')}.gif")
        with open(gif_path, "wb") as f:
            f.write(gif_bytes)
        image_paths.append(gif_path)

        return image_paths

    def get_single_preview(
        self,
        file_path: str,
        var_name: str = "Z",
    ) -> Optional[bytes]:
        frame = self.dataloader.load_single_file(file_path)
        if frame is None or var_name not in frame["data"]:
            return None

        return self.renderer.render_single(
            frame["data"][var_name],
            timestamp=frame["timestamp"],
            radar_id=frame["radar_id"],
            title=f"{var_name} - 观测",
        )

    def get_prediction_preview(
        self,
        file_paths: List[str],
        satellite_file_paths: Optional[List[str]] = None,
        frame_index: int = 0,
    ) -> Optional[bytes]:
        frames = self.dataloader.load_from_files(file_paths)
        if not frames:
            return None

        input_tensor = self.tensor_builder.build_input_tensor(frames)
        if input_tensor is None:
            return None

        satellite_tensor = None
        hard_mask_tensor = None

        if satellite_file_paths and self.use_satellite:
            satellite_frames = []
            for f in satellite_file_paths:
                try:
                    data = self.fy4_parser.parse_file(f)
                    satellite_frames.append(data)
                except Exception:
                    continue
            if satellite_frames:
                preprocessed = self.fy4_preprocessor.preprocess_for_fusion(
                    satellite_frames,
                    target_seq_len=self.tensor_builder.input_seq_len,
                    reference_times=[f["timestamp"] for f in frames],
                )
                satellite_tensor = preprocessed.get("satellite_tensor")
                hard_mask_tensor = preprocessed.get("hard_mask_tensor")

        try:
            output_tensor = self._safe_predict(input_tensor, satellite_tensor, hard_mask_tensor)
        except RuntimeError:
            return None

        denormalized = self.tensor_builder.denormalize_output(output_tensor, var_name="Z")

        if frame_index < 0 or frame_index >= len(denormalized):
            return None

        interval = self.config.get("prediction", {}).get("interval_minutes", 10)
        start_time = frames[-1]["timestamp"] + timedelta(minutes=interval)
        ts = start_time + timedelta(minutes=frame_index * interval)

        return self.renderer.render_single(
            denormalized[frame_index],
            timestamp=ts,
            radar_id=frames[0]["radar_id"],
            title=f"T+{(frame_index + 1) * interval}min - 预测",
        )
