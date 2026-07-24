import torch
import torch.nn as nn
from models.resnet50 import ResNet50Backbone
from models.msf import MultiScaleFusion
from models.cbam import CBAM

class SoftmaxHead(nn.Module):
    def __init__(self, in_features: int = 2048, num_classes: int = 4):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

class Model2_MSF_CBAM_ResNet50(nn.Module):
    """
    Model 2 / Proposed Model matching saved checkpoint weights state_dict keys perfectly.
    """
    def __init__(self, num_classes: int = 4, pretrained: bool = True):
        super().__init__()
        self.backbone = ResNet50Backbone(pretrained=pretrained)
        self.fusion = MultiScaleFusion(1024, 2048)
        self.attention = CBAM(2048)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = SoftmaxHead(2048, num_classes)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        layer3_out, layer4_out = self.backbone(x)
        x_fused = self.fusion(layer3_out, layer4_out)
        x_att = self.attention(x_fused)
        feat = torch.flatten(self.pool(x_att), 1)
        logits = self.head(feat)
        
        if return_features:
            return logits, feat
        return logits

def get_model2(num_classes: int = 4, pretrained: bool = True) -> nn.Module:
    return Model2_MSF_CBAM_ResNet50(num_classes=num_classes, pretrained=pretrained)
