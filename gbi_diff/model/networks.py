from typing import List

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Linear, Sequential, Module


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

        self._final_activation_cls = getattr(nn, final_activation) if final_activation is not None else None 
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
        if len(architecture) is None:
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
            final_activation=activation_func
        )
        self._latent_mlp = FeedForwardNetwork(
            input_dim=2 * latent_dim,
            output_dim=1,
            architecture=latent_mlp,
            activation_function=activation_func,
            final_activation=final_activation
        )

    def forward(self, theta: Tensor, x_target: Tensor) -> Tensor:
        """_summary_

        Args:
            theta (Tensor): (batch_size, theta_dim)
            x_target (Tensor): (batch_size, n_target, simulator_dim)

        Returns:
            Tensor: (batch_size, n_target, 1)
        """
        theta_enc = self._theta_encoder.forward(theta)        
        simulator_out_enc = self._simulator_out_encoder.forward(x_target)

        # repeat the theta  encoding along the n_target dimension
        n_target = x_target.shape[1]
        theta_repeat_dim = (1, n_target, 1)
        theta_enc = theta_enc[:, None].repeat(theta_repeat_dim)

        res = self._latent_mlp(torch.cat([theta_enc, simulator_out_enc], dim=-1))
        return res
