"""
TrusthOCTAI Evidential Dirichlet Head Module
Computes Dirichlet parameters alpha = Softplus(logits) + 1.0 for subjective logic uncertainty.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class EvidentialHead(nn.Module):
    def __init__(self, in_features: int = 2048, num_classes: int = 4):
        super().__init__()
        self.num_classes = num_classes
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        logits = self.fc(x)
        evidence = F.softplus(logits) + 1e-5
        alpha = evidence + 1.0
        return alpha

def get_evidence_metrics(alpha: torch.Tensor):
    """
    Computes expected probabilities and epistemic uncertainty from Dirichlet alpha parameters.
    """
    S = torch.sum(alpha, dim=1, keepdim=True)
    probs = alpha / S
    K = alpha.size(1)
    uncertainties = K / S.squeeze(1)
    return probs, uncertainties
