from abc import ABC, abstractmethod
from typing import Tuple
from torch import Tensor, nn
import torch


class Schedule(ABC):
    r"""Abstract noise schedule."""

    def __init__(self):
        super().__init__()
        self.beta_schedule: BetaSchedule

    @abstractmethod
    def forward(self, x_0: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
        """get mean and variance

        Args:
            x_0 (Tensor): vector you want to noise (batch_size, n_features) or (n_features,)
            t (Tensor): interpolation vector between 0 and diffusion time T in index space (batch_size,)

        Returns:
            Tuple[Tensor, Tensor]: noised x and noise ((batch_size), n_features)
        """
        raise NotImplementedError


class BetaSchedule(ABC):
    """Abstract Beta Schedule class"""

    def __init__(self, T: int):
        super().__init__()
        self.T = int(T)

        self._betas: Tensor
        self._alphas: Tensor
        self._alpha_bars: Tensor

    def forward(self, t: int | Tensor) -> float:
        """get beta values

        Args:
            t (int | Tensor): tensor of integers to get beta at at a certain index between 0 and T - 1

        Returns:
            Tensor: (n_timesteps, n_features)
        """
        return self._betas[t]

    def get_alphas(self, t: Tensor) -> Tensor:
        return self._alphas[t]

    def get_alpha_bar(self, t: Tensor) -> Tensor:
        return self._alpha_bars[t]


class LinearSchedule(BetaSchedule):
    def __init__(self, start: float, stop: float, T: int):
        """init class

        Args:
            start (float): start of interpolation (n_features, )
            stop (float): end of interpolation (n_features, )
            T (int): diffusion steps in the pipeline.
        """
        super().__init__(T)
        self.start = start
        self.stop = stop

        self._betas = torch.linspace(self.start, self.stop, self.T)
        self._alphas = 1 - self._betas
        self._alpha_bars = torch.cumprod(self._alphas, dim=0)


class VPSchedule(Schedule):
    def __init__(
        self,
        beta_start: float,
        beta_end: float,
        T: int,
        beta_schedule_cls: type[BetaSchedule] = LinearSchedule,
        *args,
        **kwargs
    ):
        """TODO @Julius: this is not really variance preserving but rather pushing the variance towards 1

        Args:
            beta_start (float): start beta value. If float -> handle each feature dim the same. If tensor only allowed (n_features, ) of x_0 in forward
            beta_end (float): end beta value. If float -> handle each feature dim the same. If tensor only allowed (n_features, ) of x_0 in forward
            beta_schedule_cls (BetaSchedule, optional): _description_. Defaults to LinearSchedule.
        """
        super().__init__(*args, **kwargs)

        self.beta_start = beta_start
        self.beta_end = beta_end
        self.T = T
        self.beta_schedule = beta_schedule_cls(self.beta_start, self.beta_end, self.T)

    def forward(self, x_0: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
        """get mean and variance for the variance preserving schedule

        Args:
            x_0 (Tensor): vector you want to noise (batch_size, n_features) or (n_features,)
            t (Tensor): interpolation vector between 0 and 1 (n_timesteps, )

        Returns:
            Tuple[Tensor, Tensor]: mean and variance ((batch_size), n_timesteps, n_features)
        """
        raise NotImplementedError
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


class DDPMSchedule(Schedule):
    def __init__(
        self,
        beta_start: float,
        beta_end: float,
        T: int,
        beta_schedule_cls: type[BetaSchedule] = LinearSchedule,
        *args,
        **kwargs
    ):
        """
        Args:
            beta_start (float): start beta value. If float -> handle each feature dim the same. If tensor only allowed (n_features, ) of x_0 in forward
            beta_end (float): end beta value. If float -> handle each feature dim the same. If tensor only allowed (n_features, ) of x_0 in forward
            T (int): number of diffusion steps
            beta_schedule_cls (BetaSchedule, optional): _description_. Defaults to LinearSchedule.
        """
        super().__init__(*args, **kwargs)

        self.beta_start = beta_start
        self.beta_end = beta_end
        self.T = T
        self.beta_schedule = beta_schedule_cls(self.beta_start, self.beta_end, self.T)

    def forward(self, x_0, t):
        noise = torch.normal(0, 1, size=x_0.shape)
        alpha_bar = self.beta_schedule.get_alpha_bar(t)[:, None]
        noised_x = torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1 - alpha_bar) * noise

        return noised_x, noise
