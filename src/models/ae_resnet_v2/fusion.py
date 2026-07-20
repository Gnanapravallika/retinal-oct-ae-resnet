import torch
import torch.nn as nn

class AdaptiveFeatureFusion(nn.Module):
    """
    Implements a learnable weighted multi-scale feature fusion.
    Bridges spatial detail (Layer 3) and semantic detail (Layer 4) dynamically.
    """
    def __init__(self, in_planes_l3: int = 1024, in_planes_l4: int = 2048):
        super(AdaptiveFeatureFusion, self).__init__()
        # Project spatial dimensions (14x14 -> 7x7) and channels (1024 -> 2048)
        self.projection = nn.Sequential(
            nn.Conv2d(in_planes_l3, in_planes_l4, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(in_planes_l4)
        )
        self.relu = nn.ReLU(inplace=True)
        # Learnable weight initialized to 0.5
        self.alpha = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))

    def forward(self, x3: torch.Tensor, x4: torch.Tensor) -> torch.Tensor:
        x3_down = self.projection(x3)
        # Learnable convex combination
        fused = self.alpha * x4 + (1.0 - self.alpha) * x3_down
        return self.relu(fused)
