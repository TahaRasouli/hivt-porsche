from typing import Any, Callable, Optional
import torch
from torchmetrics import Metric


class FDE(Metric):
    """
    Final Displacement Error (FDE) for trajectory prediction.
    Computes the L2 distance between predicted and target positions at the final timestep.
    """

    def __init__(self):
        super().__init__()
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        final_pred = pred[:, :, -1] if pred.dim() == 4 else pred[:, -1]  # [F, N, 2] or [N, 2]
        final_target = target[:, -1]  # [N, 2]
        self.sum += torch.norm(final_pred - final_target, p=2, dim=-1).sum()
        self.count += final_target.size(0)

    def compute(self) -> torch.Tensor:
        """Return final displacement error."""
        return self.sum / self.count
