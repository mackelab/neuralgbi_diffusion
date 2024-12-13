from abc import ABC, abstractmethod
from typing import Tuple
from torch import Tensor, nn
import torch


class Schedule(ABC):
    r"""Abstract noise schedule."""

    @abstractmethod
    def forward(self, x_0: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
        """get mean and variance

        Args:
            x_0 (Tensor): vector you want to noise (batch_size, n_features) or (n_features,)
            t (Tensor): interpolation vector between 0 and 1 (n_timesteps, )

        Returns:
            Tuple[Tensor, Tensor]: mean and variance ((batch_size), n_timesteps, n_features)
        """
        raise NotImplementedError


class BetaSchedule(ABC):
    """Abstract Beta Schedule class"""

    @abstractmethod
    def forward(self, t: Tensor) -> Tensor:
        """get beta values

        Args:
            t (Tensor): has to between 0 and 1. 0 means start and 1 means end(n_timesteps, )

        Returns:
            Tensor: (n_timesteps, n_features)
        """
        raise NotImplementedError


class LinearSchedule(BetaSchedule):
    def __init__(self, start: Tensor | float, stop: Tensor | float):
        """init class

        Args:
            start (Tensor | float): start of interpolation (n_features, )
            stop (Tensor | float): end of interpolation (n_features, )
        """

        self.start = start if isinstance(start, Tensor) else torch.tensor([start])
        self.start = self.start[None]  # shape: (1, n_features)
        self.stop = stop if isinstance(stop, Tensor) else torch.tensor([stop])
        self.stop = self.stop[None]  # shape: (1, n_features)

        self.delta = self.stop - self.start

    def forward(self, t):
        interpolation = self.start + t[:, None] * self.delta
        return interpolation


class VPSchedule(Schedule):
    def __init__(
        self,
        beta_min: Tensor | float,
        beta_max: Tensor | float,
        beta_schedule_cls: type[BetaSchedule] = LinearSchedule,
        *args,
        **kwargs
    ):
        """TODO @Julius: this is not really variance preserving but rather pushing the variance towards 1

        Args:
            beta_min (Tensor | float): start beta value. If float -> handle each feature dim the same. If tensor only allowed (n_features, ) of x_0 in forward
            beta_max (Tensor | float): end beta value. If float -> handle each feature dim the same. If tensor only allowed (n_features, ) of x_0 in forward
            beta_schedule_cls (BetaSchedule, optional): _description_. Defaults to LinearSchedule.
        """
        super().__init__(*args, **kwargs)

        self.beta_min = (
            beta_min if isinstance(beta_min, Tensor) else torch.tensor([beta_min])
        )
        self.beta_max = (
            beta_max if isinstance(beta_max, Tensor) else torch.tensor([beta_max])
        )
        self.beta_schedule = beta_schedule_cls(self.beta_min, self.beta_max)

    def forward(self, x_0: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
        """get mean and variance for the variance preserving schedule

        Args:
            x_0 (Tensor): vector you want to noise (batch_size, n_features) or (n_features,)
            t (Tensor): interpolation vector between 0 and 1 (n_timesteps, )

        Returns:
            Tuple[Tensor, Tensor]: mean and variance ((batch_size), n_timesteps, n_features)
        """
        batched = True
        if len(x_0.shape) == 1:
            x_0 = x_0[None]
            batched = False

        exp_factor = torch.exp(
            -0.5 * torch.cumsum(self.beta_schedule.forward(t), dim=0)
        )
        mean = x_0[:, None] * exp_factor
        variance = 1 - torch.square(exp_factor)
        std_dev = torch.sqrt(variance)

        batch_size, n_features = x_0.shape
        if not batched:
            mean = mean[0]
            std_dev = std_dev.repeat(1, n_features)
        else:
            std_dev = std_dev[None].repeat(batch_size, 1, n_features)

        return mean, std_dev
