# This file was copied from: https://github.com/mackelab/neuralgbi/blob/main/gbi/benchmark/tasks/linear_gaussian/task.py
# Additional slight modifications where applied

import torch
from torch import zeros, eye, randn, Tensor
from torch.distributions import MultivariateNormal


class LinearGaussianSimulator:
    def __init__(
        self,
        dim: int = 10,
        seed: int = 0,
    ):
        """Suggested beta: [1, 10, 100]"""
        _ = torch.manual_seed(seed)
        self.prior_mean = zeros((dim,))
        self.prior_cov = eye(dim)
        self.prior = MultivariateNormal(self.prior_mean, self.prior_cov)

        self.likelihood_shift = randn((dim,))
        self.likelihood_cov = torch.abs(randn((dim,))) * eye(dim)

    def simulate(self, theta: Tensor) -> Tensor:
        """Simulator."""
        chol_factor = torch.linalg.cholesky(self.likelihood_cov)
        return (
            self.likelihood_shift
            + theta
            + torch.mm(chol_factor, torch.randn_like(theta).T).T
        )