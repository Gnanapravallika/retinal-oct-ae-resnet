"""
TrusthOCTAI ResNet-50 Dual-Layer Feature Extractor
Extracts Layer 3 (1024 ch) and Layer 4 (2048 ch) features for Multi-Scale Fusion.
"""
import torch
import torch.nn as nn
import torchvision.models as models

class ResNet50Backbone(nn.Module):
    """
    ResNet-50 Backbone that extracts features from Layer 3 (spatial resolution 14x14)
    and Layer 4 (spatial resolution 7x7) to support Multi-Scale Feature Fusion.
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        backbone = models.resnet50(weights=weights)
        
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        
        self.out_channels_l3 = 1024
        self.out_channels_l4 = 2048
        
    def forward(self, x: torch.Tensor):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x3 = self.layer3(x)
        x4 = self.layer4(x3)
        return x3, x4
