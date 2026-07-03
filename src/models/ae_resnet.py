import torch
import torch.nn as nn
import torchvision.models as models

class ChannelAttention(nn.Module):
    """
    Channel Attention Module ('What' stream) that pools inputs via GAP and GMP.
    """
    def __init__(self, in_planes: int, ratio: int = 16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    """
    Spatial Attention Module ('Where' stream) that pools along channels.
    """
    def __init__(self, kernel_size: int = 7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_concat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv1(x_concat))

class ChannelSpatialAttention(nn.Module):
    """
    Combines Channel and Spatial Attention sequentially to gate feature maps.
    """
    def __init__(self, in_planes: int, ratio: int = 16):
        super(ChannelSpatialAttention, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x

class AEResNet(nn.Module):
    """
    Attention-Enhanced ResNet (AE-ResNet) classifier.
    Integrates CSA modules after Layer 3 and Layer 4 terminal blocks of ResNet-50.
    """
    def __init__(self, num_classes: int = 7, pretrained: bool = True):
        super(AEResNet, self).__init__()
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
        
        self.csa3 = ChannelSpatialAttention(in_planes=1024)
        self.csa4 = ChannelSpatialAttention(in_planes=2048)
        self.avgpool = backbone.avgpool
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(2048, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.csa3(self.layer3(x))
        x = self.csa4(self.layer4(x))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

def get_model_architecture(model_name: str, num_classes: int = 7, pretrained: bool = False) -> nn.Module:
    """
    Factory function to retrieve standard baseline networks or proposed AE-ResNet model.
    """
    model_name = model_name.lower()
    if model_name == "ae-resnet":
        return AEResNet(num_classes=num_classes, pretrained=pretrained)
    elif model_name == "resnet50":
        backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        backbone.fc = nn.Linear(backbone.fc.in_features, num_classes)
        return backbone
    elif model_name == "densenet121":
        backbone = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT if pretrained else None)
        backbone.classifier = nn.Linear(backbone.classifier.in_features, num_classes)
        return backbone
    elif model_name == "efficientnet-b0":
        backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        backbone.classifier[1] = nn.Linear(backbone.classifier[1].in_features, num_classes)
        return backbone
    elif model_name == "vit":
        backbone = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT if pretrained else None)
        backbone.heads.head = nn.Linear(backbone.heads.head.in_features, num_classes)
        return backbone
    else:
        raise ValueError(f"Unknown model architecture configuration: {model_name}")
