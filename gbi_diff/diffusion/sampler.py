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
        p: float,
        aggregation="mean",
    ):
        super().__init__(aggregation)
        assert p >= 0 and p <= 1, "p has to be a probability"
        self.p = p  # prob a loss at a diffusion time is sampled to count into the loss
        
    def forward(self, x):
        n_diffusion_steps = x.shape[1]
        # round up the number of samples -> you get the same number of samples per batch element so you can stack it.
        n_samples = np.ceil(n_diffusion_steps * self.p)
        n_samples = int(n_samples)
        samples = np.random.choice(n_diffusion_steps, size=n_samples)
        return x[:, samples]
