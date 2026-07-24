"""
TrusthOCTAI Unified Model Assembly Module
Supports exclusively the 3 core models:
  1. resnet50 (Baseline)
  2. msf_resnet50 (ResNet-50 + MSF)
  3. msf_cbam_resnet50 (ResNet-50 + MSF + CBAM Proposed)
"""
import torch
import torch.nn as nn
from models.resnet50 import ResNet50Backbone
from models.msf import MultiScaleFusion
from models.cbam import CBAM

class TrustOCTModel(nn.Module):
    """
    Modular TrustOCT Network supporting strictly the 3 core models:
      - Model 1: resnet50
      - Model 2: msf_resnet50
      - Model 3: msf_cbam_resnet50
    """
    def __init__(
        self,
        backbone_name: str = "resnet50",
        feature_module: str = "multiscale",
        attention_module: str = "cbam",
        num_classes: int = 4,
        pretrained: bool = True
    ):
        super().__init__()
        # 1. Dual-layer ResNet-50 Backbone
        self.backbone = ResNet50Backbone(pretrained=pretrained)
        in_planes = self.backbone.out_channels_l4
        
        # 2. Multi-Scale Feature Fusion
        self.feature_module_name = feature_module.lower()
        if self.feature_module_name == "multiscale":
            self.fusion = MultiScaleFusion(
                in_channels_l3=self.backbone.out_channels_l3,
                out_channels_l4=self.backbone.out_channels_l4
            )
        else:
            self.fusion = nn.Identity()
            
        # 3. CBAM Dual Attention Module
        self.attention_module_name = attention_module.lower()
        if self.attention_module_name == "cbam":
            self.attention = CBAM(in_planes)
        else:
            self.attention = nn.Identity()
            
        # 4. Standard Classifier Head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_planes, num_classes)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        # 1. Feature Extraction (x3: 1024-ch, x4: 2048-ch)
        x3, x4 = self.backbone(x)
        
        # 2. Multi-Scale Feature Fusion
        if self.feature_module_name == "multiscale":
            x_fused = self.fusion(x3, x4)
        else:
            x_fused = x4
            
        # 3. CBAM Attention Gating
        x_att = self.attention(x_fused)
        
        # 4. Global Average Pooling & Classification
        feat = self.avgpool(x_att)
        feat = torch.flatten(feat, 1)
        logits = self.fc(feat)
        
        if return_features:
            return logits, feat
            
        return logits

def build_model(config: dict) -> nn.Module:
    """
    Assembles a TrustOCTModel from configuration dictionary for the 3 models.
    """
    model_cfg = config.get("model", {})
    return TrustOCTModel(
        backbone_name=model_cfg.get("backbone", "resnet50"),
        feature_module=model_cfg.get("feature_module", "multiscale"),
        attention_module=model_cfg.get("attention", "cbam"),
        num_classes=model_cfg.get("num_classes", 4),
        pretrained=model_cfg.get("pretrained", True)
    )
