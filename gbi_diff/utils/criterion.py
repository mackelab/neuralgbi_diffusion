from typing import Tuple
from matplotlib import pyplot as plt
from matplotlib.axis import Axis
from matplotlib.figure import Figure
import numpy as np
from sklearn.metrics import r2_score
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
        self._pred: Tensor
        """(batch_size, n_target)"""
        self._d: Tensor
        """"""

    def forward(self, pred: Tensor, x: Tensor, x_target: Tensor) -> Tensor:
        """_summary_

        Args:
            pred (Tensor): (batch_size, n_target, 1)
            x (Tensor): (batch_size, n_sim_features)
            x_target (Tensor): (batch_size, n_target, n_sim_features)

        Returns:
            Tensor: loss
        """
        # distance matrix
        d = self.sample_distance(x, x_target)
        squared_distance = torch.float_power(pred[..., None] - d, 2)

        # for logging and processing
        self._pred = pred.squeeze()
        self._d = d
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

    def get_sample_correlation(self) -> Tensor:
        """computes Pearson correlation between network prediction and  distance matrix per sample in batchsize

        Returns:
            Tensor: (batch_size)
        """
        x = torch.stack([self._pred, self._d])
        x = x - x.mean(axis=-1, keepdim=True)
        cov = torch.bmm(x.permute(1, 0, 2), x.permute(1, 2, 0))

        denominator = cov[:, 0, 1]
        numerator = torch.sqrt(cov[:, 0, 0] * cov[:, 1, 1])
        corr = denominator / numerator
        return corr

    def get_correlation(self) -> Tensor:
        """get correlation of all samples

        Returns:
            Tensor: correlation for all samples
        """
        corr_mat = torch.corrcoef(torch.stack([self._d.flatten(), self._pred.flatten()]))
        return corr_mat[0, 1]
    

    @property
    def pred(self) -> Tensor:
        return self._pred
    
    @property
    def d(self) -> Tensor:
        return self._d
