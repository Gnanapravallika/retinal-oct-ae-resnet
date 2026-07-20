import torch
import torch.nn as nn

class AdaptiveFeatureFusion(nn.Module):
    """
    Adaptive Multi-Scale Fusion (AMSF) Module.
    Learns the constrained channel-wise relative contribution of mid-level spatial features (Layer 3)
    and high-level semantic features (Layer 4) during training, avoiding fixed weighting.
    Uses a sigmoid gate to constrain channel weights strictly between 0 and 1.
    """
    def __init__(self, in_planes_l3: int = 1024, in_planes_l4: int = 2048):
        super(AdaptiveFeatureFusion, self).__init__()
        # Project spatial dimensions (14x14 -> 7x7) and channels (1024 -> 2048)
        self.projection = nn.Sequential(
            nn.Conv2d(in_planes_l3, in_planes_l4, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(in_planes_l4)
        )
        self.relu = nn.ReLU(inplace=True)
        
        # Learnable channel-wise weight parameters initialized to 0.0
        # Shape: (1, 2048, 1, 1) enables broadcasting across batch and spatial dimensions
        self.beta = nn.Parameter(torch.zeros((1, in_planes_l4, 1, 1), dtype=torch.float32))

    def forward(self, x3: torch.Tensor, x4: torch.Tensor) -> torch.Tensor:
        x3_down = self.projection(x3)
        # Apply sigmoid to constrain weights strictly between 0.0 and 1.0 (sigmoid(0.0) = 0.5)
        alpha = torch.sigmoid(self.beta)
        # Channel-wise learnable convex combination
        fused = alpha * x4 + (1.0 - alpha) * x3_down
        return self.relu(fused)
