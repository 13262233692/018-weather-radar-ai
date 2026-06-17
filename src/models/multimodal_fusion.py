"""
多模态特征融合桥接模块

连接雷达数据分支和卫星数据分支，在不破坏原有 ConvLSTM 架构的前提下，
将 FY-4 卫星红外特征作为硬注意力掩码注入到每个时间步的雷达特征中。

融合模式:
  1. HARD_MASK: 直接使用卫星温度生成的硬注意力掩码点乘雷达特征
  2. GATED_FUSION: 使用可学习的卷积门控融合两者特征
  3. RESIDUAL_FUSION: 残差连接方式融合

数据流:
  Radar Frames (T, C_r, H, W)
      ↓
  [Original ConvLSTM Encoder Path] ←───┐
      ↓                                │
  Hidden States                     Attention Mask
      ↓                                ↑
  [Temporal Fusion] ←─── Satellite Features (T, C_s, H, W)
      ↓
  ConvLSTM Decoder
      ↓
  Prediction Output
"""
from typing import Optional
import torch
import torch.nn as nn

from .attention import (
    SpatialAttention,
    HardAttentionMask,
    ConvGatedFusion,
    TemporalAttentionFusion,
)


class FusionMode:
    HARD_MASK = "hard_mask"
    GATED = "gated"
    RESIDUAL = "residual"
    CONCAT = "concat"


class MultimodalFusionBridge(nn.Module):
    """
    多模态融合桥接器 - 在 ConvLSTM 编码器和解码器之间注入卫星注意力特征

    特点:
      1. 不修改原有 ConvLSTM 网络内部结构，作为附加分支独立存在
      2. 在每个时间步对雷达特征施加卫星注意力
      3. 支持可插拔的多种融合策略
    """

    def __init__(
        self,
        radar_channels: int = 2,
        satellite_channels: int = 2,
        hidden_dim: int = 64,
        fusion_mode: str = FusionMode.HARD_MASK,
    ):
        super(MultimodalFusionBridge, self).__init__()

        self.radar_channels = radar_channels
        self.satellite_channels = satellite_channels
        self.fusion_mode = fusion_mode

        self.hard_attention = HardAttentionMask(
            threshold=220.0,
            learnable=True,
            min_weight=0.1,
            max_weight=3.0,
        )

        self.spatial_attention = SpatialAttention(
            in_channels=satellite_channels,
            reduction_ratio=8,
            kernel_sizes=(3, 5, 7),
        )

        if fusion_mode == FusionMode.GATED:
            self.gated_fusion = ConvGatedFusion(
                radar_channels=radar_channels,
                satellite_channels=satellite_channels,
                hidden_channels=hidden_dim,
            )
        elif fusion_mode == FusionMode.RESIDUAL:
            self.satellite_proj = nn.Conv2d(satellite_channels, radar_channels, kernel_size=1)
        elif fusion_mode == FusionMode.CONCAT:
            self.concat_proj = nn.Conv2d(
                radar_channels + satellite_channels, radar_channels, kernel_size=3, padding=1
            )

    def forward(
        self,
        radar_sequence: torch.Tensor,
        satellite_sequence: torch.Tensor,
        hard_mask_sequence: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, C_r, H, W = radar_sequence.shape
        _, T_s, C_s, _, _ = satellite_sequence.shape

        if T_s != T:
            satellite_sequence = self._align_temporal(satellite_sequence, T)
            if hard_mask_sequence is not None:
                hard_mask_sequence = self._align_temporal(hard_mask_sequence, T)

        if self.fusion_mode == FusionMode.HARD_MASK:
            return self._hard_mask_fusion(
                radar_sequence, satellite_sequence, hard_mask_sequence
            )
        elif self.fusion_mode == FusionMode.GATED:
            return self._gated_fusion(radar_sequence, satellite_sequence, hard_mask_sequence)
        elif self.fusion_mode == FusionMode.RESIDUAL:
            return self._residual_fusion(radar_sequence, satellite_sequence)
        elif self.fusion_mode == FusionMode.CONCAT:
            return self._concat_fusion(radar_sequence, satellite_sequence)
        else:
            return self._hard_mask_fusion(
                radar_sequence, satellite_sequence, hard_mask_sequence
            )

    def _hard_mask_fusion(
        self,
        radar_sequence: torch.Tensor,
        satellite_sequence: torch.Tensor,
        hard_mask_sequence: Optional[torch.Tensor],
    ) -> torch.Tensor:
        B, T, _, H, W = radar_sequence.shape
        enhanced_frames = []

        for t in range(T):
            radar_t = radar_sequence[:, t, :, :, :]
            satellite_t = satellite_sequence[:, t, :, :, :]

            if hard_mask_sequence is not None:
                mask_t = hard_mask_sequence[:, t, :, :, :]
            else:
                tbb_t = satellite_t[:, 0:1, :, :]
                mask_t = self.hard_attention(tbb_t)

            attn_weights, _ = self.spatial_attention(satellite_t, attention_mask=mask_t)
            enhanced_t = radar_t * attn_weights

            enhanced_frames.append(enhanced_t)

        return torch.stack(enhanced_frames, dim=1)

    def _gated_fusion(
        self,
        radar_sequence: torch.Tensor,
        satellite_sequence: torch.Tensor,
        hard_mask_sequence: Optional[torch.Tensor],
    ) -> torch.Tensor:
        B, T, _, H, W = radar_sequence.shape
        enhanced_frames = []

        for t in range(T):
            radar_t = radar_sequence[:, t, :, :, :]
            satellite_t = satellite_sequence[:, t, :, :, :]

            if hard_mask_sequence is not None:
                mask_t = hard_mask_sequence[:, t, :, :, :]
            else:
                tbb_t = satellite_t[:, 0:1, :, :]
                mask_t = self.hard_attention(tbb_t)

            _, enhanced_sat = self.spatial_attention(satellite_t, attention_mask=mask_t)
            fused_t, _ = self.gated_fusion(radar_t, enhanced_sat)

            enhanced_frames.append(fused_t)

        return torch.stack(enhanced_frames, dim=1)

    def _residual_fusion(
        self,
        radar_sequence: torch.Tensor,
        satellite_sequence: torch.Tensor,
    ) -> torch.Tensor:
        B, T, _, H, W = radar_sequence.shape
        enhanced_frames = []

        for t in range(T):
            radar_t = radar_sequence[:, t, :, :, :]
            satellite_t = satellite_sequence[:, t, :, :, :]

            sat_proj = self.satellite_proj(satellite_t)
            fused_t = radar_t + sat_proj

            enhanced_frames.append(fused_t)

        return torch.stack(enhanced_frames, dim=1)

    def _concat_fusion(
        self,
        radar_sequence: torch.Tensor,
        satellite_sequence: torch.Tensor,
    ) -> torch.Tensor:
        B, T, _, H, W = radar_sequence.shape
        enhanced_frames = []

        for t in range(T):
            radar_t = radar_sequence[:, t, :, :, :]
            satellite_t = satellite_sequence[:, t, :, :, :]

            concatenated = torch.cat([radar_t, satellite_t], dim=1)
            fused_t = self.concat_proj(concatenated)

            enhanced_frames.append(fused_t)

        return torch.stack(enhanced_frames, dim=1)

    @staticmethod
    def _align_temporal(
        sequence: torch.Tensor,
        target_len: int,
    ) -> torch.Tensor:
        B, T, C, H, W = sequence.shape

        if T == target_len:
            return sequence

        if T == 1:
            return sequence.expand(B, target_len, C, H, W)

        indices = torch.linspace(0, T - 1, target_len, device=sequence.device)
        idx_floor = torch.floor(indices).long()
        idx_ceil = torch.ceil(indices).long()
        idx_ceil = torch.clamp(idx_ceil, 0, T - 1)

        t = (indices - idx_floor).view(1, -1, 1, 1, 1)

        floor_data = sequence[:, idx_floor, :, :, :]
        ceil_data = sequence[:, idx_ceil, :, :, :]

        aligned = floor_data * (1 - t) + ceil_data * t
        return aligned


class MultimodalConvLSTM(nn.Module):
    """
    多模态 ConvLSTM - 整合雷达和卫星双分支输入的完整预测模型

    架构:
      Radar Input (T, C_r, H, W) ──┐
                                     ├─> [Fusion Bridge] ──> ConvLSTM ──> Output
      Satellite Input (T, C_s, H, W) ─┘
    """

    def __init__(
        self,
        radar_channels: int = 2,
        satellite_channels: int = 2,
        hidden_channels: list = None,
        kernel_size: int = 3,
        num_layers: int = 3,
        input_seq_len: int = 24,
        output_seq_len: int = 12,
        img_height: int = 256,
        img_width: int = 256,
        fusion_mode: str = FusionMode.HARD_MASK,
        use_satellite: bool = True,
    ):
        super(MultimodalConvLSTM, self).__init__()

        if hidden_channels is None:
            hidden_channels = [64, 64, 64]

        self.use_satellite = use_satellite
        self.radar_channels = radar_channels
        self.satellite_channels = satellite_channels
        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len
        self.img_height = img_height
        self.img_width = img_width

        from .convlstm_cell import ConvLSTMCell

        self.encoder_layers = nn.ModuleList()
        prev_dim = radar_channels
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
            out_channels=radar_channels,
            kernel_size=1,
        )

        if use_satellite:
            self.fusion_bridge = MultimodalFusionBridge(
                radar_channels=radar_channels,
                satellite_channels=satellite_channels,
                hidden_dim=hidden_channels[0],
                fusion_mode=fusion_mode,
            )

    def forward(
        self,
        radar_tensor: torch.Tensor,
        satellite_tensor: Optional[torch.Tensor] = None,
        hard_mask_tensor: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.use_satellite and satellite_tensor is not None:
            fused_tensor = self.fusion_bridge(
                radar_tensor, satellite_tensor, hard_mask_tensor
            )
        else:
            fused_tensor = radar_tensor

        encoder_states = self._encode(fused_tensor)
        output_tensor = self._decode(encoder_states)

        return output_tensor

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

    def get_attention_weights(
        self,
        satellite_tensor: torch.Tensor,
        hard_mask_tensor: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """获取每个时间步的注意力权重，用于可视化"""
        if not self.use_satellite:
            return None

        B, T, C, H, W = satellite_tensor.shape
        weights_list = []

        for t in range(T):
            satellite_t = satellite_tensor[:, t, :, :, :]

            if hard_mask_tensor is not None:
                mask_t = hard_mask_tensor[:, t, :, :, :]
            else:
                tbb_t = satellite_t[:, 0:1, :, :]
                mask_t = self.fusion_bridge.hard_attention(tbb_t)

            attn_weights, _ = self.fusion_bridge.spatial_attention(
                satellite_t, attention_mask=mask_t
            )
            weights_list.append(attn_weights)

        return torch.stack(weights_list, dim=1)
