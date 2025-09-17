from typing import Any, Callable, Optional
import torch
from torchmetrics import Metric


class ADE(Metric):
    """
    Average Displacement Error (ADE) for trajectory prediction.
    Computes the mean L2 distance between predicted and target trajectories.
    """

    def __init__(self):
        super().__init__()
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        """
        Args:
            pred: Tensor of shape [F, N, T, 2] or [N, T, 2], predicted trajectories
            target: Tensor of shape [N, T, 2], ground truth trajectories
        """
        l2_error = torch.norm(pred - target, p=2, dim=-1)  # [F, N, T] or [N, T]
        self.sum += l2_error.mean(dim=-1).sum()  # mean over time, sum over batch
        self.count += pred.size(1) if pred.dim() == 4 else pred.size(0)

    def compute(self) -> torch.Tensor:
        """Return average displacement error."""
        return self.sum / self.count