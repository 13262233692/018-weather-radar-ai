"""
ConvLSTM 时空预测模型 - 用于雷达回波短临预报

输入: 过去2小时的雷达扫描序列 (24帧, 5分钟间隔)
输出: 未来2小时的雷达回波预测 (12帧, 10分钟间隔)
"""
import os
from typing import Optional, List, Tuple
import torch
import torch.nn as nn

from .convlstm_cell import ConvLSTMCell
from .multimodal_fusion import MultimodalConvLSTM, FusionMode


class ConvLSTMModel(nn.Module):
    def __init__(
        self,
        input_channels: int = 2,
        hidden_channels: List[int] = None,
        kernel_size: int = 3,
        num_layers: int = 3,
        input_seq_len: int = 24,
        output_seq_len: int = 12,
        img_height: int = 256,
        img_width: int = 256,
    ):
        super(ConvLSTMModel, self).__init__()

        if hidden_channels is None:
            hidden_channels = [64, 64, 64]

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len
        self.img_height = img_height
        self.img_width = img_width

        self.encoder_layers = nn.ModuleList()
        prev_dim = input_channels
        for i in range(num_layers):
            cell = ConvLSTMCell(
                input_dim=prev_dim,
                hidden_dim=hidden_channels[i],
                kernel_size=kernel_size,
            )
            self.encoder_layers.append(cell)
            prev_dim = hidden_channels[i]

        self.decoder_layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = hidden_channels[i - 1] if i > 0 else hidden_channels[-1]
            cell = ConvLSTMCell(
                input_dim=in_dim,
                hidden_dim=hidden_channels[i],
                kernel_size=kernel_size,
            )
            self.decoder_layers.append(cell)

        self.output_conv = nn.Conv2d(
            in_channels=hidden_channels[-1],
            out_channels=input_channels,
            kernel_size=1,
        )

    def _encode(self, input_tensor: torch.Tensor):
        batch_size = input_tensor.size(0)
        device = input_tensor.device
        image_size = (self.img_height, self.img_width)

        hidden_states = []
        for layer in self.encoder_layers:
            h, c = layer.init_hidden(batch_size, image_size, device)
            hidden_states.append((h, c))

        for t in range(self.input_seq_len):
            x = input_tensor[:, t, :, :, :]
            for layer_idx, layer in enumerate(self.encoder_layers):
                h, c = hidden_states[layer_idx]
                h, c = layer(x, (h, c))
                hidden_states[layer_idx] = (h, c)
                x = h

        return hidden_states

    def _decode(self, encoder_states: list):
        batch_size = encoder_states[0][0].size(0)
        device = encoder_states[0][0].device
        image_size = (self.img_height, self.img_width)

        decoder_states = []
        for i, layer in enumerate(self.decoder_layers):
            h, c = encoder_states[i]
            decoder_states.append((h, c))

        outputs = []
        x = encoder_states[-1][0]

        for t in range(self.output_seq_len):
            for layer_idx, layer in enumerate(self.decoder_layers):
                h, c = decoder_states[layer_idx]
                h, c = layer(x, (h, c))
                decoder_states[layer_idx] = (h, c)
                x = h

            output_frame = torch.sigmoid(self.output_conv(x))
            outputs.append(output_frame)

        output_tensor = torch.stack(outputs, dim=1)
        return output_tensor

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        encoder_states = self._encode(input_tensor)
        output_tensor = self._decode(encoder_states)
        return output_tensor


class WeatherRadarPredictor:
    def __init__(self, config: dict = None, device: Optional[str] = None):
        self.config = config or {}
        model_cfg = self.config.get("model", {})

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

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

        self.checkpoint_path = model_cfg.get("checkpoint_path", "./checkpoints/convlstm_weather.pth")
        self._load_checkpoint()

    def _load_checkpoint(self):
        if os.path.exists(self.checkpoint_path):
            try:
                checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
                if "model_state_dict" in checkpoint:
                    self.model.load_state_dict(checkpoint["model_state_dict"])
                else:
                    self.model.load_state_dict(checkpoint)
                self.model.eval()
            except Exception:
                self.model.eval()
        else:
            self.model.eval()

    @torch.no_grad()
    def predict(self, input_tensor: torch.Tensor) -> torch.Tensor:
        input_tensor = input_tensor.to(self.device)
        self.model.eval()
        output = self.model(input_tensor)
        return output.cpu()

    def save_checkpoint(self, path: str = None, epoch: int = 0, optimizer_state: dict = None):
        save_path = path or self.checkpoint_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "config": self.config,
        }
        if optimizer_state is not None:
            checkpoint["optimizer_state_dict"] = optimizer_state

        torch.save(checkpoint, save_path)


class MultimodalWeatherPredictor:
    def __init__(self, config: dict = None, device: Optional[str] = None):
        self.config = config or {}
        model_cfg = self.config.get("model", {})
        fusion_cfg = self.config.get("fusion", {})

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

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

        self.checkpoint_path = model_cfg.get("checkpoint_path", "./checkpoints/convlstm_weather.pth")
        self.multimodal_checkpoint_path = fusion_cfg.get(
            "checkpoint_path", "./checkpoints/multimodal_convlstm.pth"
        )
        self._load_checkpoint()

    def _load_checkpoint(self):
        if os.path.exists(self.multimodal_checkpoint_path):
            try:
                checkpoint = torch.load(self.multimodal_checkpoint_path, map_location=self.device)
                if "model_state_dict" in checkpoint:
                    self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
                else:
                    self.model.load_state_dict(checkpoint, strict=False)
                self.model.eval()
                return
            except Exception:
                pass

        if os.path.exists(self.checkpoint_path):
            try:
                checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
                if "model_state_dict" in checkpoint:
                    state_dict = checkpoint["model_state_dict"]
                else:
                    state_dict = checkpoint

                encoder_keys = {}
                for k, v in state_dict.items():
                    if "encoder_layers" in k or "decoder_layers" in k or "output_conv" in k:
                        encoder_keys[k] = v

                self.model.load_state_dict(encoder_keys, strict=False)
                self.model.eval()
            except Exception:
                self.model.eval()
        else:
            self.model.eval()

    @torch.no_grad()
    def predict(
        self,
        radar_tensor: torch.Tensor,
        satellite_tensor: Optional[torch.Tensor] = None,
        hard_mask_tensor: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        radar_tensor = radar_tensor.to(self.device)
        if satellite_tensor is not None:
            satellite_tensor = satellite_tensor.to(self.device)
        if hard_mask_tensor is not None:
            hard_mask_tensor = hard_mask_tensor.to(self.device)

        self.model.eval()
        output = self.model(radar_tensor, satellite_tensor, hard_mask_tensor)
        return output.cpu()

    @torch.no_grad()
    def get_attention_map(
        self,
        satellite_tensor: torch.Tensor,
        hard_mask_tensor: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        if not self.use_satellite:
            return None

        satellite_tensor = satellite_tensor.to(self.device)
        if hard_mask_tensor is not None:
            hard_mask_tensor = hard_mask_tensor.to(self.device)

        self.model.eval()
        return self.model.get_attention_weights(satellite_tensor, hard_mask_tensor).cpu()

    def save_checkpoint(self, path: str = None, epoch: int = 0, optimizer_state: dict = None):
        save_path = path or self.multimodal_checkpoint_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "config": self.config,
            "fusion_mode": self.fusion_mode,
        }
        if optimizer_state is not None:
            checkpoint["optimizer_state_dict"] = optimizer_state

        torch.save(checkpoint, save_path)
