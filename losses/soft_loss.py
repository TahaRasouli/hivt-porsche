import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftTargetCrossEntropyLoss(nn.Module):
    """
    Cross-entropy loss for soft targets (probability distributions).

    Args:
        reduction: Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, reduction: str = 'mean') -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predictions of shape [..., C], where C is number of classes.
            target: Soft targets of the same shape, representing probabilities.
        Returns:
            Tensor: Loss value (scalar if reduction is 'mean' or 'sum').
        """
        # Compute soft cross-entropy
        cross_entropy = torch.sum(-target * F.log_softmax(pred, dim=-1), dim=-1)

        if self.reduction == 'mean':
            return cross_entropy.mean()
        elif self.reduction == 'sum':
            return cross_entropy.sum()
        elif self.reduction == 'none':
            return cross_entropy
        else:
            raise ValueError(f"{self.reduction} is not a valid reduction. Use 'none', 'mean', or 'sum'.")
