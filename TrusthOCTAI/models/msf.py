"""
TrusthOCTAI Multi-Scale Feature Fusion (MSF) Module
Combines mid-level x3 (1024 ch) and deep x4 (2048 ch) feature maps.
"""
import torch
import torch.nn as nn

class MultiScaleFusion(nn.Module):
    """
    Multi-Scale Feature Fusion Module.
    Fuses mid-level x3 (1024 ch) and deep x4 (2048 ch) feature representations.
    """
    def __init__(self, in_channels_l3: int = 1024, out_channels_l4: int = 2048):
        super().__init__()
        self.conv_l3_proj = nn.Sequential(
            nn.Conv2d(in_channels_l3, out_channels_l4, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels_l4),
            nn.ReLU(inplace=True)
        )
        self.spatial_down = nn.AdaptiveAvgPool2d((7, 7))
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(out_channels_l4, out_channels_l4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels_l4),
            nn.ReLU(inplace=True)
        )

    def forward(self, x3: torch.Tensor, x4: torch.Tensor) -> torch.Tensor:
        x3_proj = self.conv_l3_proj(x3)
        x3_down = self.spatial_down(x3_proj)
        out = self.fusion_conv(x3_down + x4)
        return out
