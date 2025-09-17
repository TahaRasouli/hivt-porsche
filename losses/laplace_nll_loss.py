import torch
import torch.nn as nn


class LaplaceNLLLoss(nn.Module):
    """
    Negative log-likelihood loss assuming a Laplace distribution.

    Args:
        eps: Minimum value for scale to avoid division by zero.
        reduction: Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, eps: float = 1e-6, reduction: str = 'mean') -> None:
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Tensor of shape [..., 2*D], where the last dimension contains
                  [loc, scale] for each predicted value.
            target: Tensor of shape [..., D], ground truth values.
        Returns:
            Tensor: Loss value (scalar if reduction is 'mean' or 'sum').
        """
        # Split predictions into location and scale
        loc, scale = pred.chunk(2, dim=-1)
        scale = scale.clamp(min=self.eps)  # Avoid division by zero

        # Compute Laplace negative log-likelihood
        nll = torch.log(2 * scale) + torch.abs(target - loc) / scale

        if self.reduction == 'mean':
            return nll.mean()
        elif self.reduction == 'sum':
            return nll.sum()
        elif self.reduction == 'none':
            return nll
        else:
            raise ValueError(f"{self.reduction} is not a valid reduction. Use 'none', 'mean', or 'sum'.")
