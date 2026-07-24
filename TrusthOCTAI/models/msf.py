import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleFusion(nn.Module):
    """
    Multi-Scale Feature Fusion matching saved weights:
    conv_l3: 1024 -> 2048 (1x1 conv) added to layer4 (2048 ch) via bilinear interpolation.
    """
    def __init__(self, in_channels_l3: int = 1024, out_channels_l4: int = 2048):
        super().__init__()
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels_l3, out_channels_l4, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels_l4),
            nn.ReLU(inplace=True)
        )

    def forward(self, layer3_out: torch.Tensor, layer4_out: torch.Tensor) -> torch.Tensor:
        h, w = layer4_out.shape[2], layer4_out.shape[3]
        layer3_up = F.interpolate(layer3_out, size=(h, w), mode='bilinear', align_corners=False)
        return self.fusion_conv(layer3_up) + layer4_out
