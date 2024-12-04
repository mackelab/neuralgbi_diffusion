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
    def __init__(
        self,
        theta_dim: int,
        simulator_out_dim: int,
        latent_dim: int = 256,
        theta_encoder: List[int] = [256],
        simulator_encoder: List[int] = [256],
        latent_mlp: List[int] = [256, 256, 128],
        activation_func: str = "ELU",
        final_activation: str = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self._theta_encoder = FeedForwardNetwork(
            input_dim=theta_dim,
            output_dim=latent_dim,
            architecture=theta_encoder,
            activation_function=activation_func,
            final_activation=activation_func,
        )
        self._simulator_out_encoder = FeedForwardNetwork(
            input_dim=simulator_out_dim,
            output_dim=latent_dim,
            architecture=simulator_encoder,
            activation_function=activation_func,
            final_activation=activation_func,
        )
        self._latent_mlp = FeedForwardNetwork(
            input_dim=2 * latent_dim,
            output_dim=1,
            architecture=latent_mlp,
            activation_function=activation_func,
            final_activation=final_activation,
        )

    def forward(self, theta: Tensor, x_target: Tensor) -> Tensor:
        """

        Args:
            theta (Tensor): (batch_size, (diff_steps, ) theta_dim)
            x_target (Tensor): (batch_size, n_target, simulator_dim)

        Returns:
            Tensor: (batch_size, (diff_steps, ) n_target, 1)
        """
        batch = True
        if len(theta.shape) == 1 and len(x_target.shape) == 2:
            # input without batchsize
            batch = False
            theta = theta[None]
            x_target = x_target[None]

        # out shape: (batch_size, (diff_steps, ), latent_dim / 2)
        theta_enc = self._theta_encoder.forward(theta)  
        # out shape: (batch_size, n_target, latent_dim / 2)
        simulator_out_enc = self._simulator_out_encoder.forward(x_target)

        # repeat the theta  encoding along the n_target dimension
        n_target = x_target.shape[1]
        theta_enc = dim_repeat(theta_enc, n_target, -2)
        
        if len(theta_enc.shape) == 4:
            # diffusion steps in theta are apparent
            diffusion_steps = theta_enc.shape[1]
            simulator_out_enc = dim_repeat(simulator_out_enc, diffusion_steps, 1)

        latent_input = torch.cat([theta_enc, simulator_out_enc], dim=-1)
        res = self._latent_mlp(latent_input)

        if not batch:
            # remove artificial batch
            res = res[0]

        return res
