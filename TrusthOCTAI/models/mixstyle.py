"""
TrusthOCTAI Mid-Level MixStyle Domain Generalization Module
Mixes mean and variance feature statistics between mini-batch samples to randomize scanner noise.
"""
import random
import torch
import torch.nn as nn

class MixStyle(nn.Module):
    def __init__(self, p: float = 0.5, alpha: float = 0.1, eps: float = 1e-6):
        super().__init__()
        self.p = p
        self.alpha = alpha
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or random.random() > self.p:
            return x

        B = x.size(0)
        if B < 2:
            return x

        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()

        x_normed = (x - mu) / sig

        perm = torch.randperm(B)
        mu2, sig2 = mu[perm], sig[perm]

        lmda = torch.distributions.Beta(self.alpha, self.alpha).sample((B, 1, 1, 1)).to(x.device)

        mu_mix = mu * lmda + mu2 * (1 - lmda)
        sig_mix = sig * lmda + sig2 * (1 - lmda)

        return x_normed * sig_mix + mu_mix
