import torch
import torch.nn as nn

class ClassificationHead(nn.Module):
    """
    Modular classification head integrating Global Average Pooling, Dropout, and a Linear classifier.
    """
    def __init__(self, in_features: int = 2048, num_classes: int = 7, dropout_p: float = 0.5):
        super(ClassificationHead, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
