import numpy as np
from torch import Tensor
import torch
from abc import ABC, abstractmethod


class DiffSampler(ABC):
    def __init__(self, aggregation: str = "mean"):
        self.aggregation = getattr(
            torch, aggregation
        )  # aggregation method over diffusion steps

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """aggregates diffusion time and aggregates

        Args:
            x (Tensor): (batch_size, diffusion_time, ...)

        Returns:
            Tensor: aggregated loss tensor (batch_size, n_subsamples, ...)
        """
        pass

    def forward_unbatched(self, x: Tensor) -> Tensor:
        """aggregates diffusion time and aggregates

        Args:
            x (Tensor): (diffusion_time, ...)

        Returns:
            Tensor: aggregated loss tensor (n_subsamples, ...)
        """
        return self.forward(x[None])[0]

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)


class UniformSampler(DiffSampler):
    def __init__(
        self,
        n_samples: float,
        aggregation="mean",
    ):
        super().__init__(aggregation)
        assert isinstance(n_samples, int), "n_samples has to be an integer"
        self.n_samples = n_samples  # prob a loss at a diffusion time is sampled to count into the loss

    def forward(self, x):
        n_diffusion_steps = x.shape[1]
        assert self.n_samples < n_diffusion_steps, "Not able to sample more diffusion steps than provided without replacement"
        # round up the number of samples -> you get the same number of samples per batch element so you can stack it.
        samples_idx = np.random.choice(n_diffusion_steps, size=self.n_samples, replace=False)
        return x[:, samples_idx]
