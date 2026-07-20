import torch
import torch.nn as nn
from .backbone import ResNet50Backbone
from .fusion import AdaptiveFeatureFusion
from .attention import ChannelSpatialAttention
from .classifier import ClassificationHead

class AEResNetV2(nn.Module):
    """
    AE-ResNet v2 (Journal Edition)
    Integrates modular backbone, adaptive feature fusion, and CSA gating.
    """
    def __init__(self, num_classes: int = 7, pretrained: bool = True):
        super(AEResNetV2, self).__init__()
        self.backbone = ResNet50Backbone(pretrained=pretrained)
        self.fusion = AdaptiveFeatureFusion(in_planes_l3=1024, in_planes_l4=2048)
        self.attention = ChannelSpatialAttention(in_planes=2048)
        self.classifier = ClassificationHead(in_features=2048, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x3, x4 = self.backbone(x)
        x_fused = self.fusion(x3, x4)
        x_att = self.attention(x_fused)
        out = self.classifier(x_att)
        return out
