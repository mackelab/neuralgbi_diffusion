# This file was copied from: https://github.com/mackelab/neuralgbi/blob/main/gbi/benchmark/tasks/linear_gaussian/task.py
# Additional slight modifications where applied

from typing import Tuple, Union

import torch
from torch import ones, Tensor
from sbi.utils import BoxUniform


class UniformNoise1DSimulator:
    def __init__(
        self,
        prior_bounds: Tuple = (-1.5, 1.5),
        seed: int = 0,
        poly_coeffs: Tensor = Tensor([0.1627, 0.9073, -1.2197, -1.4639, 1.4381]),
        epsilon: Union[Tensor, float] = 0.25,
    ):
        """Suggested beta: [4, 20, 100]"""
        # Set seed.
        _ = torch.manual_seed(seed)

        # Set uniform noise half-width.
        self.epsilon = epsilon

        # Make prior for theta.
        self.prior = BoxUniform(prior_bounds[0] * ones(1), prior_bounds[1] * ones(1))

        # noise_likelihood model.
        self.noise_likelihood = BoxUniform(-self.epsilon * ones(1), self.epsilon * ones(1))

        # Set polynomial coefficients; default makes good curve.
        self.poly_coeffs = poly_coeffs

    def simulate_noiseless(self, theta: Tensor) -> Tensor:
        """Noiseless simulator."""
        return (
            torch.hstack([(0.8 * (theta + 0.25)) ** i for i in range(5)])
            * self.poly_coeffs
        ).sum(1, keepdim=True)

    def simulate(self, theta: Tensor) -> Tensor:
        """Simulator with U[-eps, eps] noise applied."""
        # Get uniform noise of [-epsilon, epsilon].
        if theta.ndim == 1:
            theta = theta.unsqueeze(0)
        noise = self.noise_likelihood.sample((theta.shape[0],))  # type: ignore
        return self.simulate_noiseless(theta) + noise
