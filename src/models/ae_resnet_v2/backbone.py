import torch
import torch.nn as nn
import torchvision.models as models

class ResNet50Backbone(nn.Module):
    """
    Wraps the ResNet-50 feature extraction backbone.
    Exposes layer outputs for multi-scale feature fusion.
    """
    def __init__(self, pretrained: bool = True):
        super(ResNet50Backbone, self).__init__()
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

    def forward(self, x: torch.Tensor) -> tuple:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x3, x4
