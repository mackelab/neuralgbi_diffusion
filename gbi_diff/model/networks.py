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
        architecture: List[int] = [],
        activation_function: str = "ReLU",
        device: str = "cpu",
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._input_dim = input_dim
        self._output_dim = output_dim

        self._activation_function_type = getattr(nn, activation_function)
        self._linear = self._create_linear_unit(architecture).to(device)

    def _create_linear_unit(self, architecture: List[int]) -> Sequential:
        """creates a linear unit specified with architecture and self._activation_function_type

        Args:
            architecture (List[int]): dimension of linear layers

        Returns:
            Sequential: sequential linear unit
        """
        # input layer
        if len(architecture) == 0:
            return Linear(self._input_dim, self._output_dim)

        layers = [
            Linear(self._input_dim, int(architecture[0])),
            self._activation_function_type(),
        ]
        # add hidden layers
        for idx in range(len(architecture) - 1):
            layers.extend(
                [
                    Linear(int(architecture[idx]), int(architecture[idx + 1])),
                    self._activation_function_type(),
                ]
            )
        # output layer
        layers.append(Linear(architecture[-1], self._output_dim))
        sequence = Sequential(*layers)
        return sequence

    def forward(self, x: Tensor):
        return self._linear(x)


class SBINetwork(Module):
    def __init__(
        self,
        theta_dim: int,
        simulator_out_dim: int,
        latent_dim: int = 256,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self._theta_encoder = FeedForwardNetwork(theta_dim, latent_dim, [256])
        self._simulator_out_encoder = FeedForwardNetwork(
            simulator_out_dim, latent_dim, [256]
        )
        self._collector = FeedForwardNetwork(2 * latent_dim, 1, [256, 256, 128])

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
        theta_repeat_dim = (1, simulator_out_enc.shape[1], 1)
        theta_enc = theta_enc[:, None].repeat(theta_repeat_dim)

        res = self._collector(torch.cat([theta_enc, simulator_out_enc], dim=-1))
        return res