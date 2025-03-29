from typing import Callable, Literal
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from gbi_diff.utils.metrics import batch_correlation, mmd_dist


class SBICriterion:
    def __init__(
        self,
        distance_func: Literal["mse", "rmse", "mmd"],
    ):
        if distance_func == "mse":
            self.distance_func = self.mse_distance
        elif distance_func == "rmse":
            self.distance_func = self.rmse_distance
        elif distance_func == "mmd":
            self.distance_func = self.mmd_distance
        else:
            raise ValueError(f"No implemented distance function for: {distance_func=}")
        
        self._pred: Tensor
        """(batch_size, n_target)"""
        self._d: Tensor
        """(batch_size, n_target)"""

        self.mse = nn.MSELoss()

    def forward(self, pred: Tensor, x: Tensor, x_target: Tensor) -> Tensor:
        """_summary_

        Args:
            pred (Tensor): (batch_size, n_target, 1) | (batch_size, n_diffusion_steps, n_target, 1)
            x (Tensor): (batch_size, n_sim_features)
            x_target (Tensor): (batch_size, n_target, n_sim_features)

        Returns:
            Tensor: loss
        """
        # distance matrix
        d = self.distance_func(x, x_target)
        d_target = d

        loss = self.mse.forward(pred[..., 0], d_target)

        # for logging and processing
        self._pred = pred[..., 0]
        self._d = d
        return loss

    @staticmethod
    def mse_distance(x: Tensor, x_target: Tensor) -> Tensor:
        """compute L2 distance

        Args:
            x (Tensor): (batch_size, n_sim_features)
            x_target (Tensor): (batch_size, n_target, n_sim_features)

        Returns:
            Tensor: (batch_size, n_target)
        """

        # L2 distance
        mse = torch.square(x[:, None] - x_target).sum(dim=-1)
        return mse
    
    @staticmethod
    def rmse_distance(x: Tensor, x_target: Tensor) -> Tensor:
        """compute L2 distance

        Args:
            x (Tensor): (batch_size, n_sim_features)
            x_target (Tensor): (batch_size, n_target, n_sim_features)

        Returns:
            Tensor: (batch_size, n_target)
        """

        # L2 distance
        mse = torch.linalg.norm(x[:, None] - x_target, dim=-1)
        return mse
    

    @staticmethod
    def mmd_distance(x: Tensor, x_target: Tensor) -> Tensor:
        """compute L2 distance

        Args:
            x (Tensor): (batch_size, n_sim_features)
            x_target (Tensor): (batch_size, n_target, n_sim_features)

        Returns:
            Tensor: (batch_size, n_target)
        """
        batch_size = len(x)
        res = torch.stack([mmd_dist(x_target[i], x[i]) for i in range(batch_size)])
        return res

    def get_sample_correlation(self) -> Tensor:
        """computes Pearson correlation between network prediction and  distance matrix per sample in batchsize

        Returns:
            Tensor: (batch_size, )
        """
        if len(self._pred.shape) == 3 and len(self._d.shape) == 3:
            # repeat d at the diffusion time axis
            # use a for loop to save memory. this requires more computation time.
            n_diff_steps = self._pred.shape[1]
            acc = 0
            for pred_idx in range(n_diff_steps):
                acc += batch_correlation(self._pred[:, pred_idx], self._d[:, 0])
            corr = acc / n_diff_steps
        else:
            corr = batch_correlation(self._pred, self._d)

        return corr

    def get_correlation(self) -> Tensor:
        """get correlation of all samples

        Returns:
            Tensor: correlation for all samples
        """
        corr_mat = torch.corrcoef(
            torch.stack([self._d.flatten(), self._pred.flatten()])
        )
        return corr_mat[0, 1]

    @property
    def pred(self) -> Tensor:
        return self._pred

    @property
    def d(self) -> Tensor:
        return self._d


class DiffusionCriterion:
    def __init__(self):
        self._mse = nn.MSELoss()

    def forward(self, pred: Tensor, noise: Tensor):
        return self._mse.forward(pred, noise)
