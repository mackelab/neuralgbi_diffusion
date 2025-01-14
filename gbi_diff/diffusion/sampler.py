import logging
import numpy as np
from torch import Tensor
import torch
from abc import ABC, abstractmethod


class DiffSampler(ABC):
    def __init__(
        self, n_samples: int, aggregation: str = "mean", on_batch_level: bool = False
    ):
        assert isinstance(n_samples, int), "n_samples has to be an integer"
        self.n_samples = n_samples  # how many samples you would like to sample
        self.aggregation = getattr(
            torch, aggregation
        )  # aggregation method over diffusion steps

        if on_batch_level and self.n_samples > 1:
            logging.warning(
                "Not able to set on_batch_level==True while n_samples > 1. Setting on_batch_level==False"
            )
            on_batch_level = False
        self.on_batch_level = on_batch_level
        # in case you get a batch of times and the number of

    @abstractmethod
    def forward(self, x: Tensor, sorted: bool = False) -> Tensor:
        """aggregates diffusion time and aggregates

        Args:
            x (Tensor): (batch_size, diffusion_time, ...)
            sorted (bool): if you would like to sort the array. In case of on_batch_level: you sort the batch dimension, otherwise the diffusion_time dimension

        Returns:
            Tensor: aggregated loss tensor (batch_size, n_subsamples, ...)
        """
        pass

    def forward_unbatched(self, x: Tensor, sorted: bool = False) -> Tensor:
        """aggregates diffusion time and aggregates

        Args:
            x (Tensor): (diffusion_time, ...)
            sorted (bool): if you would like to sort the array. In case of on_batch_level: you sort the batch dimension, otherwise the diffusion_time dimension

        Returns:
            Tensor: aggregated loss tensor (n_subsamples, ...)
        """
        return self.forward(x[None], sorted)[0]

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x, sorted=False)


class UniformSampler(DiffSampler):
    def __init__(
        self, n_samples: float, aggregation="mean", on_batch_level: bool = False
    ):
        super().__init__(n_samples, aggregation, on_batch_level)

    def forward(self, x, sorted):
        batch_size, n_diffusion_steps = x.shape
        if self.on_batch_level:
            samples_idx = np.random.choice(
                n_diffusion_steps, size=batch_size, replace=False
            )
            samples = x[samples_idx]
            sort_axis = 0
        else:
            assert (
                self.n_samples < n_diffusion_steps
            ), "Not able to sample more diffusion steps than provided without replacement"
            # round up the number of samples -> you get the same number of samples per batch element so you can stack it.

            samples_idx = np.random.choice(
                n_diffusion_steps, size=self.n_samples, replace=False
            )
            samples = np.sort(samples, axis=0)
            samples = x[:, samples_idx]
            sort_axis = 1

        if sorted:
            samples = np.sort(samples, axis=sort_axis)

        return samples
