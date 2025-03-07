from copy import deepcopy
import functools
from typing import Tuple

from einops import rearrange
from matplotlib import pyplot as plt
import numpy as np
import torch
from lightning import LightningModule
from lightning.pytorch.loggers import TensorBoardLogger
from torch import Tensor, optim

from gbi_diff.model.networks import DiffusionNetwork, SBINetwork

# from gbi_diff.utils.metrics import batch_correlation
from gbi_diff.utils.encoding import get_positional_encoding

from gbi_diff.utils.train_potential_config import _Model as ModelConfig
from gbi_diff.utils.train_potential_config import _Optimizer as OptimizerConfig

from gbi_diff.utils.train_guidance_config import _Diffusion as DiffusionGuidanceConfig
from gbi_diff.utils.train_guidance_config import _Model as ModelGuidanceConfig
from gbi_diff.utils.train_guidance_config import _Optimizer as OptimizerGuidanceConfig

from gbi_diff.utils.train_diffusion_config import _Optimizer as DiffusionOptimizerConfig
from gbi_diff.utils.train_diffusion_config import _Model as DiffusionModelConfig
from gbi_diff.utils.train_diffusion_config import _Diffusion as DiffusionDiffusionConfig

from gbi_diff.utils.criterion import DiffusionCriterion, SBICriterion
from gbi_diff.utils.plot import (
    plot_correlation,
    plot_diffusion_step_corr,
    plot_diffusion_step_loss,
)
import gbi_diff.diffusion.schedule as diffusion_schedule


class PotentialNetwork(LightningModule):
    def __init__(
        self,
        theta_dim: int,
        simulator_out_dim: int,
        optimizer_config: OptimizerConfig,
        net_config: ModelConfig,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()

        optimizer_config = deepcopy(optimizer_config)
        net_config = deepcopy(net_config)

        self._net = SBINetwork(
            theta_dim=theta_dim,
            simulator_out_dim=simulator_out_dim,
            theta_encoder=net_config.ThetaEncoder,
            simulator_encoder=net_config.SimulatorEncoder,
            latent_mlp=net_config.LatentMLP,
        )
        self.example_input_array = (
            torch.zeros(1, theta_dim),
            torch.zeros(1, 1, simulator_out_dim),
        )

        self.criterion = SBICriterion(distance_order=2)
        self._optimizer_config = optimizer_config.__dict__

        # this thing should not leave the class. Inconsistencies with strings feared
        self._train_step_outputs = {"pred": [], "d": []}
        self._val_step_outputs = {"pred": [], "d": []}

    def _batch_forward(self, batch: Tuple[Tensor, Tensor, Tensor]) -> Tensor:
        theta, simulator_out, x_target = batch
        network_res = self.forward(theta, x_target)
        loss = self.criterion.forward(network_res, simulator_out, x_target)
        return loss

    def forward(self, theta: Tensor, x_target: Tensor) -> Tensor:
        """_summary_

        Args:
            prior (Tensor): (batch_size, n_prior_features)
            x_target (Tensor): (batch_size, n_target, n_sim_features)

        Returns:
            Tensor: (batch_size, n_target)
        """
        return self._net.forward(theta, x_target)

    def training_step(self, batch: Tuple[Tensor, Tensor, Tensor], batch_idx: int):
        loss = self._batch_forward(batch)
        self.log("train/loss", loss, on_epoch=True, on_step=False)
        self.log(
            "train/cost_corr",
            self.criterion.get_sample_correlation().mean(),
            on_epoch=True,
            on_step=False,
        )

        self._train_step_outputs["pred"].append(self.criterion.pred)
        self._train_step_outputs["d"].append(self.criterion.d)

        return loss

    def on_train_epoch_end(self):
        # reset logging data
        self._train_step_outputs = {"pred": [], "d": []}

    def validation_step(self, batch: Tuple[Tensor, Tensor, Tensor], batch_idx: int):
        loss = self._batch_forward(batch)
        self.log("val/loss", loss, on_epoch=True, on_step=False)
        self.log(
            "val/cost_corr",
            self.criterion.get_sample_correlation().mean(),
            on_epoch=True,
            on_step=False,
        )

        self._val_step_outputs["pred"].append(self.criterion.pred)
        self._val_step_outputs["d"].append(self.criterion.d)

        return loss

    def on_validation_epoch_end(self):
        if self._val_step_outputs == {"pred": [], "d": []}:
            return
        pred = torch.cat(self._val_step_outputs["pred"], dim=0)
        d = torch.cat(self._val_step_outputs["d"], dim=0)
        fig, _ = plot_correlation(pred, d, agg=True)

        tb_logger: TensorBoardLogger = self.loggers[0]
        tb_logger.experiment.add_figure(
            "val/corr_plot", fig, global_step=self.global_step
        )

        plt.close(fig)
        self._val_step_outputs = {"pred": [], "d": []}

    def configure_optimizers(self):
        optimizer_cls = getattr(optim, self._optimizer_config.pop("name"))
        optimizer = optimizer_cls(self.parameters(), **self._optimizer_config)
        return optimizer


class _DiffusionBase(LightningModule):
    def __init__(self, diff_config: DiffusionGuidanceConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.period_spread = diff_config.period_spread
        self.time_repr_dim = diff_config.time_repr_dim

        self.diff_schedule = self._build_schedule(deepcopy(diff_config))
        self.diffusion_steps = diff_config.steps
        """same thing as: T (capital T)"""
        self.t = torch.linspace(0, 1, self.diffusion_steps)

        assert (
            self.diffusion_steps % 100 == 0
        ), "For validation, T has to be a multiple of 100"
        self.val_t = torch.linspace(0, self.diffusion_steps - 1, 100).int()
        self.val_t_repr = self.get_diff_time_repr(self.val_t).float()

    def _build_schedule(
        self, diff_config: DiffusionGuidanceConfig
    ) -> diffusion_schedule.Schedule:
        schedule_name = diff_config.__dict__.pop("diffusion_schedule")
        schedule_cls = getattr(diffusion_schedule, schedule_name)
        schedule_config = getattr(diff_config, schedule_name)
        schedule_config.beta_schedule_cls = getattr(
            diffusion_schedule, schedule_config.beta_schedule_cls
        )
        schedule = schedule_cls(**schedule_config.__dict__)
        return schedule

    def get_diff_time_repr(self, t: np.ndarray) -> Tensor:
        """get sinusoidal positional encoding

        Args:
            t (np.ndarray): time in index space (batch_size, )

        Returns:
            Tensor: positional encoding (batch_size, repr_dim)
        """
        return torch.from_numpy(
            get_positional_encoding(t, self.time_repr_dim, self.period_spread)
        ).float()

    def _get_diff_time_sample(self, batch_size: int) -> Tensor:
        """

        Args:
            batch_size (int): how many samples you stack ontop of each other
        Returns:
            Tensor: (batch_size, ) indices between 0 and self.diffusion_steps.
        """
        if self.training:
            sampled_t = torch.from_numpy(
                np.random.choice(self.diffusion_steps, size=batch_size, replace=True)
            )
        else:
            # always take the same subsample
            raise NotImplementedError
        return sampled_t

    def _sample_diffusions(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """sample a subset of thetas according to `self.sampler` and diffuse the samples
        according to `self.schedule`

        Args:
            x (Tensor): (batch_size, n_features)

        Returns:
            Tuple[Tensor, Tensor, Tensor]:
                - noised x (batch_size, n_features)
                - normal noise (batch_size, n_features)
                - time representations (batch_size, time_features)
        """
        batch_size, _ = x.shape

        # sample a subset of diffusion time
        sampled_t = self._get_diff_time_sample(batch_size)
        noised_x, noise = self.diff_schedule.forward(x, sampled_t)

        t_repr = self.get_diff_time_repr(sampled_t)

        return noised_x, noise, t_repr


class Guidance(_DiffusionBase):
    """time dependent version of the potential function"""

    def __init__(
        self,
        theta_dim: int,
        simulator_out_dim: int,
        optimizer_config: OptimizerGuidanceConfig,
        net_config: ModelGuidanceConfig,
        diff_config: DiffusionGuidanceConfig,
        *args,
        **kwargs
    ):
        super().__init__(diff_config, *args, **kwargs)
        self.save_hyperparameters()

        optimizer_config = deepcopy(optimizer_config)
        net_config = deepcopy(net_config)

        self._net = SBINetwork(
            theta_dim=theta_dim,
            simulator_out_dim=simulator_out_dim,
            theta_encoder=net_config.ThetaEncoder,
            simulator_encoder=net_config.SimulatorEncoder,
            time_encoder=net_config.TimeEncoder,
            latent_mlp=net_config.LatentMLP,
        )
        self.example_input_array = (
            torch.zeros(1, theta_dim),
            torch.zeros(1, 1, simulator_out_dim),
            torch.zeros(1, net_config.TimeEncoder.input_dim),
        )

        self.criterion = SBICriterion(distance_order=2)
        self._optimizer_config = optimizer_config.__dict__

        # this thing should not leave the class. Inconsistencies with strings feared
        self._train_step_outputs = {"pred": [], "d": []}
        self._val_step_outputs = {"pred": [], "d": []}

    def forward(self, theta_t: Tensor, x_target: Tensor, time_repr: Tensor) -> Tensor:
        """_summary_

        Args:
            theta (Tensor): (batch_size, theta_dim)
            x_target (Tensor): (batch_size, n_target, simulator_dim)
            time (Tensor): (batch_size, time_repr_dim). Defaults to None

        Returns:
            Tensor: (batch_size, n_target, 1)
        """
        return self._net.forward(theta_t, x_target, time_repr)

    def _batch_forward(self, batch: Tuple[Tensor, Tensor, Tensor]) -> Tensor:
        theta, simulator_out, x_target = batch

        theta_t, _, time_repr = self._sample_diffusions(theta)
        network_res = self.forward(theta_t, x_target, time_repr)
        loss = self.criterion.forward(network_res, simulator_out, x_target)
        return loss

    def training_step(self, batch, batch_idx):
        self.train()
        loss = self._batch_forward(batch)
        self.log("train/loss", loss, on_epoch=True, on_step=False)
        self.log(
            "train/cost_corr",
            self.criterion.get_sample_correlation().mean(),
            on_epoch=True,
            on_step=False,
        )

        self._train_step_outputs["pred"].append(self.criterion.pred)
        self._train_step_outputs["d"].append(self.criterion.d)
        return loss

    def on_train_epoch_end(self):
        # reset logging data
        self._train_step_outputs = {"pred": [], "d": []}

    def validation_step(self, batch, batch_idx):
        self.eval()
        theta, simulator_out, x_target = batch
        batch_size = len(theta)

        loss_acc = 0
        sample_corr = 0
        preds = []
        targets = []
        for sampled_t, t_repr in zip(self.val_t, self.val_t_repr):
            sampled_t = sampled_t.repeat(batch_size)
            t_repr = t_repr[None].repeat(batch_size, 1)
            theta_t, _ = self.diff_schedule.forward(theta, sampled_t)
            pred = self.forward(theta_t, x_target, t_repr)
            loss = self.criterion.forward(pred, simulator_out, x_target)
            loss_acc += loss
            sample_corr += self.criterion.get_sample_correlation().mean()
            preds.append(self.criterion.pred)
            targets.append(self.criterion.d)

        preds = torch.stack(preds)
        targets = torch.stack(targets)
        self._val_step_outputs["pred"].append(preds)
        self._val_step_outputs["d"].append(targets)

        self.log("val/loss", loss_acc / len(self.val_t), on_epoch=True, on_step=False)
        self.log(
            "val/cost_corr",
            sample_corr / len(self.val_t),
            on_epoch=True,
            on_step=False,
        )

        return loss

    def on_validation_epoch_end(self):
        if self._val_step_outputs == {"pred": [], "d": []}:
            return
        pred = torch.cat(self._val_step_outputs["pred"], dim=1)
        pred = rearrange(pred, "T B F -> B T F")
        d = torch.cat(self._val_step_outputs["d"], dim=1)
        d = rearrange(d, "T B F -> B T F")

        tb_logger: TensorBoardLogger = self.loggers[0]

        fig, _ = plot_diffusion_step_loss(
            pred, d, x_high=self.diffusion_steps - 1, agg=True
        )
        tb_logger.experiment.add_figure(
            "val/diff_loss_plot", fig, global_step=self.global_step
        )

        # fig, ax = plot_diffusion_step_corr(pred, d[:, 0], agg=True)
        # tb_logger.experiment.add_figure(
        #     "val/diff_corr_plot", fig, global_step=self.global_step
        # )

        plt.close(fig)

        self._val_step_outputs = {"pred": [], "d": []}

    def configure_optimizers(self):
        optimizer_cls = getattr(optim, self._optimizer_config.pop("name"))
        optimizer = optimizer_cls(self.parameters(), **self._optimizer_config)
        return optimizer


class DiffusionModel(_DiffusionBase):
    def __init__(
        self,
        theta_dim: int,
        diff_config: DiffusionDiffusionConfig,
        optimizer_config: DiffusionOptimizerConfig,
        net_config: DiffusionModelConfig,
        *args,
        **kwargs
    ):
        super().__init__(diff_config, *args, **kwargs)
        self.save_hyperparameters()

        optimizer_config = deepcopy(optimizer_config)
        net_config = deepcopy(net_config)

        self._net = DiffusionNetwork(
            theta_dim=theta_dim,
            theta_encoder=net_config.ThetaEncoder,
            time_encoder=net_config.TimeEncoder,
            latent_mlp=net_config.LatentMLP,
        )

        self.example_input_array = (
            torch.zeros(1, theta_dim),
            torch.zeros(1, net_config.TimeEncoder.input_dim),
        )

        self.criterion = DiffusionCriterion()
        self._optimizer_config = optimizer_config.__dict__

        # this thing should not leave the class. Inconsistencies with strings feared
        self._train_step_outputs = {"pred": [], "target": []}
        self._val_step_outputs = {"pred": [], "target": []}

    def forward(self, theta_t: Tensor, time_repr: Tensor) -> Tensor:
        return self._net.forward(theta_t, time_repr)

    def _batch_forward(self, batch: Tuple[Tensor, Tensor, Tensor]) -> Tensor:
        theta, _, _ = batch

        theta_t, noise, time_repr = self._sample_diffusions(theta)
        pred = self.forward(theta_t, time_repr)
        loss = self.criterion.forward(pred, noise)
        return loss

    def training_step(self, batch, batch_idx):
        self.train()
        loss = self._batch_forward(batch)
        self.log("train/loss", loss, on_epoch=True, on_step=False)
        return loss

    def on_train_epoch_end(self):
        self._train_step_outputs = {"pred": [], "target": []}

    def validation_step(self, batch, batch_idx):
        self.eval()
        theta, _, _ = batch
        batch_size = len(theta)

        loss_acc = 0
        preds = []
        targets = []
        for sampled_t, t_repr in zip(self.val_t, self.val_t_repr):
            sampled_t = sampled_t.repeat(batch_size)
            t_repr = t_repr[None].repeat(batch_size, 1)
            theta_t, noise = self.diff_schedule.forward(theta, sampled_t)
            pred = self.forward(theta_t, t_repr)
            loss = self.criterion.forward(pred, noise)
            loss_acc += loss
            preds.append(pred)
            targets.append(noise)

        preds = torch.stack(preds)
        targets = torch.stack(targets)
        self._val_step_outputs["pred"].append(preds)
        self._val_step_outputs["target"].append(targets)

        self.log("val/loss", loss_acc / len(self.val_t), on_epoch=True, on_step=False)
        return loss

    def on_validation_epoch_end(self):
        if self._val_step_outputs == {"pred": [], "target": []}:
            return
        pred = torch.cat(self._val_step_outputs["pred"], dim=1)
        pred = rearrange(pred, "T B F -> B T F")
        target = torch.cat(self._val_step_outputs["target"], dim=1)
        target = rearrange(target, "T B F -> B T F")

        tb_logger: TensorBoardLogger = self.loggers[0]
        fig, _ = plot_diffusion_step_loss(
            pred, target, x_high=self.diffusion_steps - 1, agg=True
        )
        tb_logger.experiment.add_figure(
            "val/diff_loss_plot", fig, global_step=self.global_step
        )
        plt.close(fig)
        self._val_step_outputs = {"pred": [], "target": []}

    def configure_optimizers(self):
        optimizer_cls = getattr(optim, self._optimizer_config.pop("name"))
        optimizer = optimizer_cls(self.parameters(), **self._optimizer_config)
        return optimizer
