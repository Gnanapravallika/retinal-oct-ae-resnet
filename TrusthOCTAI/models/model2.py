"""
Model 2 Architecture: MSF + CBAM + ResNet-50 (Standard Softmax)
Peak In-Domain Feature Extractor Backbone (96.17% Test Accuracy).
"""
import torch
import torch.nn as nn
from models.resnet50 import ResNet50Backbone
from models.msf import MultiScaleFusion
from models.cbam import CBAM

class Model2_MSF_CBAM_ResNet50(nn.Module):
    """
    Model 2 Architecture:
      1. Backbone: ResNet-50 (Extracts x3: 1024-ch, x4: 2048-ch)
      2. Feature Module: Multi-Scale Feature Fusion (MSF)
      3. Attention Module: CBAM Dual Attention (Channel + Spatial)
      4. Classifier Head: Standard Linear + Softmax Head
    """
    def __init__(self, num_classes: int = 4, pretrained: bool = True):
        super().__init__()
        # 1. Dual-layer ResNet-50 Backbone
        self.backbone = ResNet50Backbone(pretrained=pretrained)
        in_planes = self.backbone.out_channels_l4  # 2048 ch
        
        # 2. Multi-Scale Feature Fusion (x3 + x4)
        self.fusion = MultiScaleFusion(
            in_channels_l3=self.backbone.out_channels_l3,
            out_channels_l4=self.backbone.out_channels_l4
        )
        
        # 3. CBAM Dual Attention
        self.attention = CBAM(in_planes)
        
        # 4. Global Average Pooling & Linear Head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_planes, num_classes)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        # 1. Extract Layer 3 (x3) and Layer 4 (x4) features
        x3, x4 = self.backbone(x)
        
        # 2. Multi-Scale Fusion
        x_fused = self.fusion(x3, x4)
        
        # 3. Apply CBAM Channel & Spatial Attention
        x_att = self.attention(x_fused)
        
        # 4. Global Pooling & Classification
        feat = self.avgpool(x_att)
        feat = torch.flatten(feat, 1)
        logits = self.fc(feat)
        
        if return_features:
            return logits, feat
            
        return logits

def get_model2(num_classes: int = 4, pretrained: bool = True) -> nn.Module:
    """
    Factory function for Model 2 (MSF + CBAM + ResNet-50).
    """
    return Model2_MSF_CBAM_ResNet50(num_classes=num_classes, pretrained=pretrained)
