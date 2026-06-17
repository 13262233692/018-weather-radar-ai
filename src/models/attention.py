"""
手动实现的 Spatial Attention 空间注意力网络层

核心思想:
  1. 从 FY-4 卫星红外云顶温度特征中提取空间注意力掩码
  2. 将掩码作为硬注意力权重，与雷达特征进行逐元素点乘融合
  3. 云顶温度越低（对流越强）→ 注意力权重越高 → 雷达特征被放大

实现要点:
  - 不依赖任何预训练网络，完全手动编写
  - 支持可学习的温度阈值和缩放因子
  - 支持硬注意力（0/1掩码）和软注意力（连续权重）
  - 支持多尺度特征提取（3×3, 5×5, 7×7）

"""
import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialAttention(nn.Module):
    """
    空间注意力层 - 从卫星红外特征中学习空间注意力掩码

    输入:
        satellite_features: (B, C_sat, H, W) - FY-4 红外特征 (IR1 + IR2)
        或 attention_mask: (B, 1, H, W) - 预计算的硬注意力掩码

    输出:
        attention_weights: (B, 1, H, W) - 归一化的空间注意力权重
        enhanced_features: (B, C_sat, H, W) - 经注意力增强的卫星特征
    """

    def __init__(
        self,
        in_channels: int = 2,
        reduction_ratio: int = 8,
        kernel_sizes: tuple = (3, 5, 7),
        use_bias: bool = True,
        temperature_init: float = 220.0,
        temperature_range: float = 100.0,
    ):
        super(SpatialAttention, self).__init__()

        self.in_channels = in_channels
        self.reduction_ratio = reduction_ratio
        self.kernel_sizes = kernel_sizes
        self.temperature_init = temperature_init
        self.temperature_range = temperature_range

        self.temperature_threshold = nn.Parameter(
            torch.tensor(temperature_init, dtype=torch.float32)
        )
        self.temperature_scale = nn.Parameter(
            torch.tensor(1.0 / temperature_range, dtype=torch.float32)
        )

        hidden_dim = max(in_channels // reduction_ratio, 1)

        self.multi_scale_convs = nn.ModuleList()
        for k in kernel_sizes:
            padding = k // 2
            conv = nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=k, padding=padding, bias=use_bias),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
            )
            self.multi_scale_convs.append(conv)

        self.fusion_conv = nn.Conv2d(
            hidden_dim * len(kernel_sizes), 1, kernel_size=1, bias=use_bias
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(
        self,
        satellite_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple:
        if attention_mask is not None:
            attention_weights = torch.sigmoid(attention_mask)
            enhanced = satellite_features * attention_weights
            return attention_weights, enhanced

        B, C, H, W = satellite_features.shape

        tbb_feature = satellite_features[:, 0:1, :, :]
        temp_diff = self.temperature_threshold - tbb_feature
        temp_weight = torch.sigmoid(temp_diff * self.temperature_scale)

        scale_features = []
        for conv in self.multi_scale_convs:
            feat = conv(satellite_features)
            scale_features.append(feat)

        multi_scale = torch.cat(scale_features, dim=1)
        spatial_logits = self.fusion_conv(multi_scale)

        spatial_attention = torch.sigmoid(spatial_logits)

        attention_weights = spatial_attention * temp_weight

        attention_weights = attention_weights + 0.5
        attention_weights = torch.clamp(attention_weights, 0.1, 3.0)

        enhanced_features = satellite_features * attention_weights

        return attention_weights, enhanced_features


class HardAttentionMask(nn.Module):
    """
    硬注意力掩码生成器 - 从 FY-4 云顶亮温中生成二值化硬注意力掩码

    对流云团中心的云顶亮温通常 < 220K，以此为阈值生成硬注意力
    """

    def __init__(
        self,
        threshold: float = 220.0,
        learnable: bool = True,
        min_weight: float = 0.1,
        max_weight: float = 3.0,
    ):
        super(HardAttentionMask, self).__init__()

        self.min_weight = min_weight
        self.max_weight = max_weight

        if learnable:
            self.threshold = nn.Parameter(torch.tensor(threshold, dtype=torch.float32))
        else:
            self.register_buffer('threshold', torch.tensor(threshold, dtype=torch.float32))

        self.slope = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def forward(self, tbb: torch.Tensor) -> torch.Tensor:
        tbb = tbb.float()

        temp_diff = self.threshold - tbb
        hard_mask = torch.sigmoid(temp_diff * self.slope)

        hard_mask = self.min_weight + hard_mask * (self.max_weight - self.min_weight)
        return hard_mask


class ChannelAttention(nn.Module):
    """
    通道注意力层 - 自适应加权不同卫星通道的贡献
    """

    def __init__(self, in_channels: int = 2, reduction_ratio: int = 4):
        super(ChannelAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        hidden_dim = max(in_channels // reduction_ratio, 1)

        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, in_channels, kernel_size=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))

        channel_weights = torch.sigmoid(avg_out + max_out)
        return channel_weights


class ConvGatedFusion(nn.Module):
    """
    卷积门控融合 - 自适应控制雷达特征和卫星特征的融合强度

    f = sigmoid(W * radar + (1 - sigmoid(W)) * satellite
    """

    def __init__(
        self,
        radar_channels: int = 2,
        satellite_channels: int = 2,
        hidden_channels: int = 16,
        kernel_size: int = 3,
    ):
        super(ConvGatedFusion, self).__init__()

        padding = kernel_size // 2

        self.gate_conv = nn.Sequential(
            nn.Conv2d(
                radar_channels + satellite_channels,
                hidden_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=True,
            ),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                hidden_channels,
                1,
                kernel_size=kernel_size,
                padding=padding,
                bias=True,
            ),
            nn.Sigmoid(),
        )

        self.radar_proj = nn.Conv2d(
            radar_channels, radar_channels, kernel_size=1, bias=False
        )
        self.satellite_proj = nn.Conv2d(
            satellite_channels, radar_channels, kernel_size=1, bias=False
        )

    def forward(
        self,
        radar_feat: torch.Tensor,
        satellite_feat: torch.Tensor,
    ) -> torch.Tensor:
        concatenated = torch.cat([radar_feat, satellite_feat], dim=1)

        gate = self.gate_conv(concatenated)

        radar_proj = self.radar_proj(radar_feat)
        satellite_proj = self.satellite_proj(satellite_feat)

        fused = gate * radar_proj + (1 - gate) * satellite_proj
        return fused, gate


class SatelliteFeatureExtractor(nn.Module):
    """
    FY-4 卫星红外特征提取网络

    从 2 通道红外数据提取多尺度深度特征，用于后续的注意力生成
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 16,
    ):
        super(SatelliteFeatureExtractor, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels * 2),
            nn.ReLU(inplace=True),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.channel_attn = ChannelAttention(in_channels=out_channels, reduction_ratio=4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat1 = self.conv1(x)
        feat2 = self.conv2(feat1)
        feat3 = self.conv3(feat2)

        ch_weights = self.channel_attn(feat3)
        enhanced = feat3 * ch_weights

        return enhanced


class TemporalAttentionFusion(nn.Module):
    """
    时序注意力融合 - 按时间步将卫星注意力掩码与雷达特征点乘融合

    对于每个时间步 t:
        F_radar_enhanced[t] = F_radar[t] * M_satellite[t]

    其中 M_satellite[t] 是 t 时刻从卫星数据生成的空间注意力掩码
    """

    def __init__(
        self,
        radar_channels: int = 2,
        satellite_channels: int = 2,
        use_gated_fusion: bool = True,
    ):
        super(TemporalAttentionFusion, self).__init__()

        self.use_gated_fusion = use_gated_fusion

        if use_gated_fusion:
            self.gated_fusion = ConvGatedFusion(
                radar_channels=radar_channels,
                satellite_channels=satellite_channels,
            )
        else:
            self.satellite_proj = nn.Conv2d(
                satellite_channels, radar_channels, kernel_size=1
            )

        self.spatial_attention = SpatialAttention(
            in_channels=satellite_channels
        )

        self.hard_attention = HardAttentionMask(threshold=220.0)

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

        enhanced_frames = []

        for t in range(T):
            radar_t = radar_sequence[:, t, :, :, :]
            satellite_t = satellite_sequence[:, t, :, :, :]

            tbb_t = satellite_t[:, 0:1, :, :]

            hard_mask_t = None
            if hard_mask_sequence is not None:
                hard_mask_t = hard_mask_sequence[:, t, :, :, :]
            else:
                hard_mask_t = self.hard_attention(tbb_t)

            attn_weights, enhanced_sat = self.spatial_attention(
                satellite_t, attention_mask=hard_mask_t
            )

            if self.use_gated_fusion:
                fused_t, _ = self.gated_fusion(radar_t, enhanced_sat)
            else:
                sat_proj = self.satellite_proj(enhanced_sat)
                fused_t = radar_t * attn_weights + sat_proj

            enhanced_frames.append(fused_t)

        enhanced_sequence = torch.stack(enhanced_frames, dim=1)
        return enhanced_sequence

    def _align_temporal(
        self,
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
