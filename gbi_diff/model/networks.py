from typing import List

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Linear, Sequential, Module

from gbi_diff.utils.reshape import dim_repeat


class FeedForwardNetwork(Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        architecture: List[int] = None,
        activation_function: str = "ReLU",
        device: str = "cpu",
        final_activation: str = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._input_dim = input_dim
        self._output_dim = output_dim

        self._final_activation_cls = (
            getattr(nn, final_activation) if final_activation is not None else None
        )
        self._activation_function_cls = getattr(nn, activation_function)
        self._linear = self._create_linear_unit(architecture).to(device)

    def _create_linear_unit(self, architecture: List[int] = None) -> Sequential:
        """creates a linear unit specified with architecture and self._activation_function_type

        Args:
            architecture (List[int]): dimension of linear layers

        Returns:
            Sequential: sequential linear unit
        """
        # input layer
        if architecture is None or (
            isinstance(architecture, list) and len(architecture) == 0
        ):
            return Linear(self._input_dim, self._output_dim)

        layers = [
            Linear(self._input_dim, int(architecture[0])),
            self._activation_function_cls(),
        ]
        # add hidden layers
        for idx in range(len(architecture) - 1):
            layers.extend(
                [
                    Linear(int(architecture[idx]), int(architecture[idx + 1])),
                    self._activation_function_cls(),
                ]
            )
        # output layer
        layers.append(Linear(architecture[-1], self._output_dim))

        if self._final_activation_cls is not None:
            layers.append(self._final_activation_cls())

        sequence = Sequential(*layers)
        return sequence

    def forward(self, x: Tensor) -> Tensor:
        return self._linear(x)


class SBINetwork(Module):
    from gbi_diff.utils.train_guidance_config import (
        _ThetaEncoder,
        _TimeEncoder,
        _SimulatorEncoder,
        _LatentMLP,
    )

    def __init__(
        self,
        theta_dim: int,
        simulator_out_dim: int,
        theta_encoder: _ThetaEncoder,
        simulator_encoder: _SimulatorEncoder,
        latent_mlp: _LatentMLP,
        time_encoder: _TimeEncoder = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self._theta_encoder = FeedForwardNetwork(
            input_dim=theta_dim,
            output_dim=theta_encoder.output_dim,
            architecture=theta_encoder.architecture,
            activation_function=theta_encoder.activation_func,
            final_activation=theta_encoder.final_activation,
        )
        self._simulator_out_encoder = FeedForwardNetwork(
            input_dim=simulator_out_dim,
            output_dim=simulator_encoder.output_dim,
            architecture=simulator_encoder.architecture,
            activation_function=simulator_encoder.activation_func,
            final_activation=simulator_encoder.final_activation,
        )

        latent_input_dim = theta_encoder.output_dim + simulator_encoder.output_dim
        if time_encoder is not None and time_encoder.enabled:
            self._time_encoder = FeedForwardNetwork(
                input_dim=time_encoder.input_dim,
                output_dim=time_encoder.output_dim,
                architecture=time_encoder.architecture,
                activation_function=time_encoder.activation_func,
                final_activation=time_encoder.final_activation,
            )
            latent_input_dim += time_encoder.output_dim

        self._latent_mlp = FeedForwardNetwork(
            input_dim=latent_input_dim,
            output_dim=1,
            architecture=latent_mlp.architecture,
            activation_function=latent_mlp.activation_func,
            final_activation=latent_mlp.final_activation,
        )

    def forward(
        self, theta: Tensor, x_target: Tensor, time_repr: Tensor = None
    ) -> Tensor:
        """
        Args:
            theta (Tensor): (batch_size, theta_dim)
            x_target (Tensor): (batch_size, n_target, simulator_dim)
            time (Tensor): (batch_size, time_repr_dim). Defaults to None

        Returns:
            Tensor: (batch_size, n_target, 1)
        """
        batch = True
        if len(theta.shape) == 1 and len(x_target.shape) == 2:
            # input without batchsize
            batch = False
            theta = theta[None]
            x_target = x_target[None]
            if time_repr is not None and len(time_repr.shape) == 1:
                time_repr = time_repr[None]

        # out shape: (batch_size, latent_dim)
        theta_enc = self._theta_encoder.forward(theta)
        # out shape: (batch_size, n_target, latent_dim)
        simulator_out_enc = self._simulator_out_encoder.forward(x_target)
        # out shape: (batch_size, latent_dim)

        # repeat the theta  encoding along the n_target dimension
        n_target = x_target.shape[1]
        theta_enc = dim_repeat(theta_enc, n_target, -2)

        # if len(theta_enc.shape) == 4:
        #     # diffusion steps in theta are apparent
        #     diffusion_steps = theta_enc.shape[1]
        #     simulator_out_enc = dim_repeat(simulator_out_enc, diffusion_steps, 1)

        latent_input = torch.cat([theta_enc, simulator_out_enc], dim=-1)

        # if existing add time representation
        if time_repr is not None:
            time_enc = self._time_encoder.forward(time_repr)
            time_enc = dim_repeat(time_enc, n_target, -2)
            latent_input = torch.cat([latent_input, time_enc], dim=-1)

        res = self._latent_mlp(latent_input)

        if not batch:
            # remove artificial batch
            res = res[0]

        return res


class DiffusionNetwork(Module):
    from gbi_diff.utils.train_diffusion_config import (
        _ThetaEncoder,
        _TimeEncoder,
        _LatentMLP,
    )

    def __init__(
        self,
        theta_dim: int,
        theta_encoder: _ThetaEncoder,
        time_encoder: _TimeEncoder,
        latent_mlp: _LatentMLP,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self._theta_encoder = FeedForwardNetwork(
            input_dim=theta_dim,
            output_dim=theta_encoder.output_dim,
            architecture=theta_encoder.architecture,
            activation_function=theta_encoder.activation_func,
            final_activation=theta_encoder.final_activation,
        )

        self._time_encoder = FeedForwardNetwork(
            input_dim=time_encoder.input_dim,
            output_dim=time_encoder.output_dim,
            architecture=time_encoder.architecture,
            activation_function=time_encoder.activation_func,
            final_activation=time_encoder.final_activation,
        )

        self._latent_mlp = FeedForwardNetwork(
            input_dim=theta_encoder.output_dim + time_encoder.output_dim,
            output_dim=theta_dim,
            architecture=latent_mlp.architecture,
            activation_function=latent_mlp.activation_func,
            final_activation=latent_mlp.final_activation,
        )

    def forward(self, theta: Tensor, time_repr: Tensor) -> Tensor:
        concat_dim = 1
        if len(theta.shape) == 1:
            concat_dim = 0

        theta_enc = self._theta_encoder.forward(theta)
        time_enc = self._time_encoder.forward(time_repr)
        latent_inp = torch.cat([theta_enc, time_enc], dim=concat_dim)
        out = self._latent_mlp.forward(latent_inp)
        return out
