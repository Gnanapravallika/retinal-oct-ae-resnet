import torch
import torch.nn as nn
from .backbone import ResNet50Backbone
from .fusion import AdaptiveFeatureFusion
from .attention import ChannelSpatialAttention
from .classifier import ClassificationHead

class AEResNetV2(nn.Module):
    """
    AE-ResNet v2 (Journal Edition)
    Highly modular architecture supporting standard residual feature extraction,
    Adaptive Multi-Scale Fusion (AMSF), and Channel-Spatial Attention (CSA) gating.
    Supports easy ablation configuration via flags.
    """
    def __init__(self, num_classes: int = 7, pretrained: bool = True, use_attention: bool = True, use_adaptive: bool = True):
        super(AEResNetV2, self).__init__()
        self.backbone = ResNet50Backbone(pretrained=pretrained)
        self.use_attention = use_attention
        self.use_adaptive = use_adaptive
        
        if self.use_adaptive:
            # AMSF (Adaptive Multi-Scale Fusion)
            self.fusion = AdaptiveFeatureFusion(in_planes_l3=1024, in_planes_l4=2048)
        else:
            # Fixed feature projection alignment
            self.projection = nn.Sequential(
                nn.Conv2d(1024, 2048, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(2048)
            )
            self.relu = nn.ReLU(inplace=True)
            
        if self.use_attention:
            self.attention = ChannelSpatialAttention(in_planes=2048)
            
        self.classifier = ClassificationHead(in_features=2048, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x3, x4 = self.backbone(x)
        
        if self.use_adaptive:
            x_fused = self.fusion(x3, x4)
        else:
            x3_down = self.projection(x3)
            x_fused = self.relu(x4 + x3_down)
            
        if self.use_attention:
            x_fused = self.attention(x_fused)
            
        out = self.classifier(x_fused)
        return out

def get_model_v2(model_name: str, num_classes: int = 7, pretrained: bool = False) -> nn.Module:
    """
    Model architecture factory wrapper.
    Returns the compiled PyTorch model matching the given model name config.
    """
    model_name_lower = model_name.lower().replace("-", "_")
    if model_name_lower == "resnet50":
        import torchvision.models as models
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    elif model_name_lower == "densenet121":
        import torchvision.models as models
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        model = models.densenet121(weights=weights)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        return model
    elif model_name_lower == "resnet_fixed_fusion":
        return AEResNetV2(num_classes=num_classes, pretrained=pretrained, use_attention=False, use_adaptive=False)
    elif model_name_lower == "ae_resnet_v1":
        return AEResNetV2(num_classes=num_classes, pretrained=pretrained, use_attention=True, use_adaptive=False)
    elif model_name_lower == "resnet_amsf":
        return AEResNetV2(num_classes=num_classes, pretrained=pretrained, use_attention=False, use_adaptive=True)
    elif model_name_lower in ["ae_resnet_v2", "ae_resnet", "ae-resnet"]:
        return AEResNetV2(num_classes=num_classes, pretrained=pretrained, use_attention=True, use_adaptive=True)
    else:
        raise ValueError(f"Unknown model name configuration: {model_name}")
