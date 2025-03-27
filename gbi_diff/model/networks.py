from typing import Callable, List, Union

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Linear, Sequential, Module

from gbi_diff.utils.metrics import compute_distances
from gbi_diff.utils.reshape import dim_repeat
from sbi.utils.sbiutils import standardizing_net
from sbi.neural_nets.embedding_nets import (
    PermutationInvariantEmbedding,
    FCEmbedding,
)
from pyknos.nflows.nn import nets


class MultiplyByMean(nn.Module):
    def __init__(self, mean: Union[Tensor, float], std: Union[Tensor, float]):
        super(MultiplyByMean, self).__init__()
        mean, std = map(torch.as_tensor, (mean, std))
        self.mean = mean
        self.std = std
        self.register_buffer("_mean", mean)
        self.register_buffer("_std", std)

    def forward(self, tensor):
        return tensor * self._mean


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
    from gbi_diff.utils.configs.train_guidance import (
        _ThetaEncoder,
        _TimeEncoder,
        _SimulatorEncoder,
        _LatentMLP,
    )

    def __init__(
        self,
        theta_dim: int,
        simulator_out_dim: int,
        trail_dim: int,
        theta_encoder: _ThetaEncoder,
        simulator_encoder: _SimulatorEncoder,
        latent_mlp: _LatentMLP,
        time_encoder: _TimeEncoder = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if simulator_encoder.enabled and trail_dim is not None and trail_dim > 1:
            # permutation invariant embedding net required
            trial_net = FCEmbedding(
                input_dim=simulator_out_dim,
                output_dim=simulator_encoder.hidden_dim,
                num_layers=2,
                num_hiddens=40,
            )
            self._sim_enc = PermutationInvariantEmbedding(
                trial_net=trial_net,
                trial_net_output_dim=simulator_encoder.hidden_dim,
                output_dim=simulator_encoder.hidden_dim,
            )
            hidden_dim = simulator_encoder.hidden_dim
        elif simulator_encoder.enabled and (trail_dim is None or trail_dim <= 1):
            self._sim_enc = FCEmbedding(
                input_dim=simulator_out_dim,
                output_dim=simulator_encoder.hidden_dim,
                num_layers=2,
                num_hiddens=40,
            )
            hidden_dim = simulator_encoder.hidden_dim
        else:
            self._sim_enc = nn.Identity()
            hidden_dim = simulator_out_dim

        if time_encoder.enabled:
            self._time_enc = FCEmbedding(
                input_dim=time_encoder.input_dim,
                output_dim=time_encoder.hidden_dim,
                num_layers=time_encoder.n_layers,
                num_hiddens=time_encoder.hidden_dim,
            )
            hidden_dim += time_encoder.hidden_dim
        else:
            self._time_enc = nn.Identity()
            hidden_dim += time_encoder.input_dim

        if theta_encoder.enabled:
            self._theta_enc = FCEmbedding(
                input_dim=theta_dim,
                output_dim=theta_encoder.hidden_dim,
                num_layers=theta_encoder.n_layers,
                num_hiddens=theta_encoder.hidden_dim,
            )
            hidden_dim += theta_encoder.hidden_dim
        else:
            self._theta_enc = nn.Identity()
            hidden_dim += theta_dim

        if latent_mlp.net_type == "res_net":
            self._latent_mlp = nets.ResidualNet(
                in_features=hidden_dim,
                out_features=1,
                hidden_features=latent_mlp.hidden_dim,
                num_blocks=latent_mlp.n_layers,
                dropout_probability=latent_mlp.dropout_prob,
                use_batch_norm=latent_mlp.use_batch_norm,
            )
        elif latent_mlp.net_type == "MLP":
            self._latent_mlp = nets.MLP(
                in_shape=[hidden_dim],
                out_shape=[1],
                hidden_sizes=[latent_mlp.hidden_dim] * latent_mlp.n_layers,
                activate_output=False,
            )
        else:
            raise ValueError(
                f"Unrecognized net type: {latent_mlp.net_type}. Available are 'res_net' and 'MLP'."
            )

        self._latent_mlp = nn.Sequential(self._latent_mlp, nn.Softplus())

        self._standardize_theta_nn = nn.Identity()
        self._standardize_x_nn = nn.Identity()
        self._dist_multiplier = nn.Identity()

    def init_standardize_net(self, theta: Tensor, x: Tensor):
        """init network standardizer for theta and x encoder

        Args:
            theta (Tensor): (batch_dim, theta_dim)
            x (Tensor): (batch_dim, x_dim)
        """
        self._standardize_theta_nn = standardizing_net(theta, False)
        self._theta_enc = nn.Sequential(self._standardize_theta_nn, self._theta_enc)

        self._standardize_x_nn = standardizing_net(x, False)
        self._sim_enc = nn.Sequential(self._standardize_x_nn, self._sim_enc)

    def init_distr_multiplier(
        self,
        x: Tensor,
        x_target: Tensor,
        distance_func: Callable[[Tensor, Tensor], Tensor],
        monte_carlo: int = None,
    ):
        """init multiplier for distribution

        Args:
            x (Tensor): (batch_dim, x_dim) or (batch_dim, n_trial, x_dim)
            x_target (Tensor): (n_target, x_dim) or (n_target, n_trial, x_dim)
            distance_func (Callable[[Tensor, Tensor], Tensor]): distance function for pairwise comparison
            monte_carlo (int, optional): if you would like to have only a rough estimate about
                the distances, set this to an integer of how many samples you would like to look at.
                Defaults to None.

        """
        if monte_carlo is not None and monte_carlo > 0:
            idx = np.random.choice(len(x_target), size=monte_carlo)
            x_target = x_target[idx]
            idx = np.random.choice(len(x), size=monte_carlo)
            x = x[idx]

        distances = compute_distances(distance_func, x_target, x)
        mean_distance = torch.mean(distances)
        std_distance = torch.std(distances)
        self._dist_multiplier = MultiplyByMean(mean_distance, std_distance)
        self._latent_mlp = nn.Sequential(
            self._latent_mlp, nn.Softplus(), self._dist_multiplier
        )

    def forward(
        self, theta: Tensor, x_target: Tensor, time_repr: Tensor = None
    ) -> Tensor:
        x_embed = self._sim_enc.forward(x_target)
        theta_embed = self._theta_enc.forward(theta)
        time_embed = self._time_enc.forward(time_repr)

        # repeat theta_embed, and time embed
        n_target = x_target.shape[1]
        theta_embed = dim_repeat(theta_embed, int(n_target), 1)
        time_embed = dim_repeat(time_embed, int(n_target), 1)

        hidden = torch.concat((theta_embed, x_embed, time_embed), dim=-1)
        res = self._latent_mlp.forward(hidden)
        return res


class SBINetwork2(Module):
    from gbi_diff.utils.configs.train_guidance import (
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
    from gbi_diff.utils.configs.train_diffusion import (
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
        """_summary_

        Args:
            theta (Tensor): (batch_size, param_dim)
            time_repr (Tensor): (batch_size, time_repr_dim)

        Returns:
            Tensor: (batch_size, param_dim)
        """
        concat_dim = 1
        if len(theta.shape) == 1:
            concat_dim = 0

        theta_enc = self._theta_encoder.forward(theta)
        time_enc = self._time_encoder.forward(time_repr)
        latent_inp = torch.cat([theta_enc, time_enc], dim=concat_dim)
        out = self._latent_mlp.forward(latent_inp)
        return out
