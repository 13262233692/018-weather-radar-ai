"""
核心业务服务编排 - 串联解析、预处理、预测、可视化全流程
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
from .models.convlstm_model import WeatherRadarPredictor
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
        self.predictor = WeatherRadarPredictor(self.config)
        self.renderer = RadarImageRenderer(self.config)

    def process_files(
        self,
        file_paths: List[str],
        save_images: bool = False,
        output_dir: str = None,
    ) -> Dict:
        frames = self.dataloader.load_from_files(file_paths)
        if not frames:
            return {"success": False, "error": "Failed to load any radar frames"}

        return self._process_frames(frames, save_images, output_dir)

    def process_bytes(
        self,
        bytes_list: List[Tuple[str, bytes]],
        save_images: bool = False,
        output_dir: str = None,
    ) -> Dict:
        frames = self.dataloader.load_from_bytes_list(bytes_list)
        if not frames:
            return {"success": False, "error": "Failed to load any radar frames"}

        return self._process_frames(frames, save_images, output_dir)

    def _process_frames(
        self,
        frames: List[dict],
        save_images: bool,
        output_dir: Optional[str],
    ) -> Dict:
        result = {
            "success": True,
            "input_frame_count": len(frames),
            "radar_id": frames[0]["radar_id"],
            "start_time": frames[0]["timestamp"],
            "end_time": frames[-1]["timestamp"],
        }

        input_tensor = self.tensor_builder.build_input_tensor(frames)
        if input_tensor is None:
            result["success"] = False
            result["error"] = (
                f"Insufficient frames. Need at least "
                f"{self.tensor_builder.input_seq_len}, got {len(frames)}"
            )
            return result

        with torch.no_grad():
            output_tensor = self.predictor.predict(input_tensor)

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
        frame_index: int = 0,
    ) -> Optional[bytes]:
        frames = self.dataloader.load_from_files(file_paths)
        if not frames:
            return None

        input_tensor = self.tensor_builder.build_input_tensor(frames)
        if input_tensor is None:
            return None

        output_tensor = self.predictor.predict(input_tensor)
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
