from typing import List, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Linear, Sequential, Module

from gbi_diff.utils.reshape import dim_repeat
from sbi.neural_nets.embedding_nets import (
    PermutationInvariantEmbedding,
    FCEmbedding,
)
from pyknos.nflows.nn import nets
from sbi.utils.sbiutils import Standardize


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


class Concatenate(nn.Module):
    def __init__(self, dim: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim = dim
        
    def forward(self, inputs: List[Tensor]) -> Tensor:
        return torch.cat(inputs, dim=self.dim)


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


class AdaMLPBlock(nn.Module):
    r"""Creates a residual MLP block module with adaptive layer norm for conditioning.

    Arguments:
        hidden_dim: The dimensionality of the MLP block.
        cond_dim: The number of embedding features.
    """

    def __init__(
        self,
        hidden_dim: int,
        cond_dim: int,
        mlp_ratio: int = 1,
    ):
        super().__init__()

        self.ada_ln = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3 * hidden_dim),
        )

        # Initialize the last layer to zero
        self.ada_ln[-1].weight.data.zero_()
        self.ada_ln[-1].bias.data.zero_()

        # MLP block
        # NOTE: This can be made more flexible to support layer types.
        self.block = nn.Sequential(
            nn.LayerNorm(hidden_dim, elementwise_affine=False),
            nn.Linear(hidden_dim, hidden_dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(hidden_dim * mlp_ratio, hidden_dim),
        )

    def forward(self, x: Tensor, yt: Tensor) -> Tensor:
        """
        Arguments:
            x: The input tensor, with shape (B, D_x).
            t: The embedding vector, with shape (B, D_t).

        Returns:
            The output tensor, with shape (B, D_x).
        """

        a, b, c = self.ada_ln(yt).chunk(3, dim=-1)

        y = (a + 1) * x + b
        y = self.block(y)
        y = x + c * y
        y = y / torch.sqrt(1 + c * c)

        return y


class AdaMLP(nn.Module):
    """
    MLP denoising network using adaptive layer normalization for conditioning.
    Relevant literature:

    See "Scalable Diffusion Models with Transformers", by William Peebles, Saining Xie.

    Arguments:
        x_dim: The dimensionality of the input tensor.
        emb_dim: The number of embedding features.
        input_handler: The input handler module.
        hidden_dim: The dimensionality of the MLP block.
        num_layers: The number of MLP blocks.
        **kwargs: Key word arguments handed to the AdaMLPBlock.
    """

    def __init__(
        self,
        x_dim: int,
        emb_dim: int,
        input_handler: nn.Module,
        hidden_dim: int = 100,
        num_layers: int = 3,
        **kwargs,
    ):
        super().__init__()
        self.input_handler = input_handler
        self.num_layers = num_layers

        self.ada_blocks = nn.ModuleList()
        for _i in range(num_layers):
            self.ada_blocks.append(AdaMLPBlock(hidden_dim, emb_dim, **kwargs))

        self.input_layer = nn.Linear(x_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, x_dim)

    def forward(self, inputs: Tuple[Tensor, Tensor, Tensor]) -> Tensor:
        x, y, t = inputs
        x, y, t = self.input_handler(x, y, t)
        yt = torch.cat([y, t], dim=-1)

        h = self.input_layer(x)
        for i in range(self.num_layers):
            h = self.ada_blocks[i](h, yt)
        return self.output_layer(h)


class AdaMLP_Scoring(nn.Module):
    """
    MLP denoising network using adaptive layer normalization for conditioning.
    Relevant literature: https://arxiv.org/abs/2212.09748

    See "Scalable Diffusion Models with Transformers", by William Peebles, Saining Xie.

    Arguments:
        x_dim: The dimensionality of the input tensor.
        emb_dim: The number of embedding features.
        input_handler: The input handler module.
        hidden_dim: The dimensionality of the MLP block.
        num_layers: The number of MLP blocks.
        **kwargs: Key word arguments handed to the AdaMLPBlock.
    """

    def __init__(
        self,
        x_dim: int,
        emb_dim: int,
        input_handler: nn.Module = None,
        hidden_dim: int = 100,
        num_layers: int = 3,
        **kwargs,
    ):
        super().__init__()

        def pass_through(*args):
            return args

        self.input_handler = (
            input_handler if input_handler is not None else pass_through
        )
        self.num_layers = num_layers

        self.ada_blocks = nn.ModuleList()
        for _i in range(num_layers):
            self.ada_blocks.append(AdaMLPBlock(hidden_dim, emb_dim, **kwargs))

        self.input_layer = nn.Linear(x_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, 1)
        # add softplus to output
        self.softplus = nn.Softplus()

    def forward(self, inputs: Tuple[Tensor, Tensor, Tensor]) -> Tensor:
        x, y, t = inputs
        yt = torch.cat([y, t], dim=-1)

        h = self.input_layer(x)
        for i in range(self.num_layers):
            h = self.ada_blocks[i](h, yt)
        return self.softplus(self.output_layer(h))


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
        x_dim: int,
        trail_dim: int,
        theta_encoder: _ThetaEncoder,
        simulator_encoder: _SimulatorEncoder,
        latent_mlp: _LatentMLP,
        time_encoder: _TimeEncoder = None,
        theta_stats: Tuple[Tensor, Tensor] = None,  # (mean, std)
        x_stats: Tuple[Tensor, Tensor] = None,  # (mean, std)
        distance_stats: Tuple[Tensor, Tensor] = None,  # (mean, std)
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.theta_dim = theta_dim
        self.x_dim = x_dim
        self.trail_dim = trail_dim
        self.theta_stats = theta_stats
        self.x_stats = x_stats
        self.distance_stats = distance_stats

        self._sim_enc, self._sim_enc_out_dim = self._build_x_encoder(simulator_encoder)
        self._theta_enc, self._theta_enc_out_dim = self._build_theta_encoder(
            theta_encoder
        )
        self._time_enc, self._time_enc_out_dim = self._build_time_encoder(time_encoder)
        hidden_dim = (
            self._sim_enc_out_dim + self._theta_enc_out_dim + self._time_enc_out_dim
        )
        self._latent_mlp = self._build_latent_mlp(latent_mlp, hidden_dim)

    def _build_x_encoder(self, config: _SimulatorEncoder) -> Tuple[nn.Module, int]:
        if self.x_stats is not None:
            net = Standardize(*self.x_stats)
        else:
            net = nn.Identity()
        sim_enc = nn.Sequential(net)

        if config.enabled and self.trail_dim is not None and self.trail_dim > 1:
            # permutation invariant embedding net required
            trial_net = FCEmbedding(
                input_dim=self.x_dim,
                output_dim=config.hidden_dim,
                num_layers=2,
                num_hiddens=40,
            )
            net = PermutationInvariantEmbedding(
                trial_net=trial_net,
                trial_net_output_dim=config.hidden_dim,
                output_dim=config.hidden_dim,
            )
            out_dim = config.hidden_dim
        elif config.enabled and (self.trail_dim is None or self.trail_dim <= 1):
            net = FCEmbedding(
                input_dim=self.x_dim,
                output_dim=config.hidden_dim,
                num_layers=2,
                num_hiddens=40,
            )
            out_dim = config.hidden_dim
        else:
            self._sim_enc = nn.Identity()
            out_dim = self.x_dim
        sim_enc = sim_enc.append(net)

        return sim_enc, out_dim

    def _build_theta_encoder(self, config: _ThetaEncoder) -> Tuple[nn.Module, int]:
        if self.theta_stats is not None:
            net = Standardize(*self.theta_stats)
        else:
            net = nn.Identity()
        theta_enc = nn.Sequential(net)

        if config.enabled:
            net = FCEmbedding(
                input_dim=self.theta_dim,
                output_dim=config.hidden_dim,
                num_layers=config.n_layers,
                num_hiddens=config.hidden_dim,
            )
            out_dim = config.hidden_dim
        else:
            net = nn.Identity()
            out_dim = self.theta_dim
        theta_enc.append(net)
        return theta_enc, out_dim

    def _build_time_encoder(self, config: _TimeEncoder) -> Tuple[nn.Module, int]:
        if config.enabled:
            time_enc = FCEmbedding(
                input_dim=config.input_dim,
                output_dim=config.hidden_dim,
                num_layers=config.n_layers,
                num_hiddens=config.hidden_dim,
            )
            out_dim = config.hidden_dim
        else:
            time_enc = nn.Identity()
            out_dim = config.input_dim
        return time_enc, out_dim

    def _build_latent_mlp(self, config: _LatentMLP, input_dim: int) -> nn.Module:
        if config.net_type == "res_net":
            net = nn.Sequential(
                Concatenate(dim=-1),
                nets.ResidualNet(
                    in_features=input_dim,
                    out_features=1,
                    hidden_features=config.hidden_dim,
                    num_blocks=config.n_layers,
                    dropout_probability=config.dropout_prob,
                    use_batch_norm=config.use_batch_norm,
                ),
            )
        elif config.net_type == "MLP":
            net = nn.Sequential(
                Concatenate(dim=-1),
                FCEmbedding(
                    input_dim=input_dim,
                    output_dim=config.hidden_dim,
                    num_layers=config.n_layers - 1,
                    num_hiddens=config.hidden_dim,
                ),
                nn.Linear(config.hidden_dim, 1),
            )
        elif config.net_type == "AdaMLP":
            net = AdaMLP_Scoring(
        x_dim=self.             _sim_enc_out_dim,
                emb_dim=input_dim - self._sim_enc_out_dim,
                input_handler=None,
                hidden_dim=config.hidden_dim,
                num_layers=config.n_layers,
            )
        else:
            raise ValueError(
                f"Unrecognized net type: {config.net_type}. Available are 'res_net', 'MLP' and 'AdaMLP."
            )
        latent_mlp = nn.Sequential(net, nn.Softplus())

        if self.distance_stats is not None:
            latent_mlp = latent_mlp.append(MultiplyByMean(*self.distance_stats))
        return latent_mlp

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

        res = self._latent_mlp.forward([theta_embed, x_embed, time_embed])
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
