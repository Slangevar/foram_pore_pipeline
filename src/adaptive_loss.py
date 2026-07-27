
import torch
import torch.nn as nn
from . import metrics

class AdaptiveTverskyCELoss(nn.Module):
    """
    Adaptive Tversky Cross-Entropy Loss with learnable weights.
    Reference: UNet-3D with Adaptive TverskyCE Loss for Pancreas Medical Image Segmentation (arXiv:2505.01951)
    
    Uses learnable weights (log variances) to balance Tversky Loss and Cross-Entropy Loss.
    Loss = (1 / (2 * sigma_t^2)) * L_tversky + log(sigma_t) + (1 / (2 * sigma_ce^2)) * L_ce + log(sigma_ce)
    """
    def __init__(self, alpha=0.3, beta=0.7, initial_sigma_t=1.0, initial_sigma_ce=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        # Initialize log variances (log(sigma^2))
        # We use log_sigma because standard deviation must be positive
        self.log_sigma_t = nn.Parameter(torch.tensor(initial_sigma_t).log())
        self.log_sigma_ce = nn.Parameter(torch.tensor(initial_sigma_ce).log())

    def forward(self, y_pred, y_true, weight=None):
        # Calculate individual losses
        l_tversky = metrics.tversky_loss(y_pred, y_true, alpha=self.alpha, beta=self.beta, weight=weight)
        
        # Calculate Cross-Entropy (handling weighted/unweighted)
        # Using existing weighted_crossentropy_loss or implementing simple CE
        # Re-using simple CE logic for consistency with metrics.py style, but robust
        l_ce = metrics.crossentropy_loss(y_pred, y_true, weight=weight)

        # Combine with learnable weights (Homoscedastic Uncertainty or simple learnable)
        # Following standard Kendall et al. formulation:
        # Loss = L_1 * exp(-log_var_1) + 0.5 * log_var_1
        # Here we use a simplified version often used: L_total = w1 * L1 + w2 * L2
        # BUT the paper says "learnable weights".
        # Let's use the explicit learnable scalar logic for transparency:
        # loss = sigma_t * L_t + sigma_ce * L_ce
        # where sigma are learnable parameters.
        # However, to avoid negative weights, we often use exp or sigmoid.
        # Let's stick to the Kendall method as it's the "Adaptive Loss" standard.
        
        # Method: Kendall et al. (Multi-Task Learning using Uncertainty)
        precision_t = torch.exp(-self.log_sigma_t)
        loss_t = precision_t * l_tversky + 0.5 * self.log_sigma_t

        precision_ce = torch.exp(-self.log_sigma_ce)
        loss_ce = precision_ce * l_ce + 0.5 * self.log_sigma_ce

        return loss_t + loss_ce
