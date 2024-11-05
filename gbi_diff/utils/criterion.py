import torch
import torch.nn as nn
from torch import Tensor
import torch.linalg as LA


class SBICriterion:
    def __init__(
        self,
        distance_order: int = 2.0,
    ):
        self._distance_order = distance_order

    def forward(self, pred: Tensor, x: Tensor, x_target: Tensor) -> Tensor:
        """_summary_

        Args:
            pred (Tensor): (batch_size, 1)
            x (Tensor): (batch_size, n_sim_features)
            x_target (Tensor): (batch_size, n_target, n_sim_features)

        Returns:
            Tensor: loss
        """
        # distance matrix
        d = self.sample_distance(x, x_target)
        squared_distance = torch.float_power(pred[..., None] - d, 2)
        # NOTE:@Julius why sum it and not taking the mean??? -> better for logging  
        squared_distance = torch.sum(squared_distance, dim=-1)
        return torch.mean(squared_distance)

    def sample_distance(self, x: Tensor, x_target: Tensor) -> Tensor:
        """compute L2 distance

        Args:
            x (Tensor): (batch_size, n_sim_features)
            x_target (Tensor): (batch_size, n_target, n_sim_features)

        Returns:
            Tensor: (batch_size, n_target)
        """
        d = x[:, None] - x_target
        distance = LA.norm(d, ord=self._distance_order, dim=-1)
        return distance
