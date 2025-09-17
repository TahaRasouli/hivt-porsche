from typing import Any, Callable, Optional
import torch
from torchmetrics import Metric

class MR(Metric):
    """
    Miss Rate (MR) for trajectory prediction.
    Computes the fraction of predicted trajectories whose final displacement
    exceeds a given miss threshold.
    """

    def __init__(self, miss_threshold: float = 2.0):
        super().__init__()
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.miss_threshold = miss_threshold

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        final_pred = pred[:, :, -1] if pred.dim() == 4 else pred[:, -1]  # [F, N, 2] or [N, 2]
        final_target = target[:, -1]  # [N, 2]
        misses = torch.norm(final_pred - final_target, p=2, dim=-1) > self.miss_threshold
        self.sum += misses.sum()
        self.count += final_target.size(0)

    def compute(self) -> torch.Tensor:
        return self.sum / self.count
