from copy import deepcopy
from typing import Tuple

from matplotlib import pyplot as plt
import torch
from lightning import LightningModule
from lightning.pytorch.loggers import TensorBoardLogger
from torch import Tensor, optim

from gbi_diff.model.networks import SBINetwork
# from gbi_diff.utils.metrics import batch_correlation
from gbi_diff.utils.train_config import _Model as ModelConfig
from gbi_diff.utils.train_config import _Optimizer as OptimizerConfig
from gbi_diff.utils.train_theta_noise_config import _Diffusion as DiffusionConfig
from gbi_diff.utils.criterion import SBICriterion
from gbi_diff.utils.plot import (
    plot_correlation,
    plot_diffusion_step_corr,
    plot_diffusion_step_loss,
)
import gbi_diff.diffusion.schedule as diffusion_schedule
import gbi_diff.diffusion.sampler as diffusion_sampler


class PotentialFunction(LightningModule):
    def __init__(
        self,
        theta_dim: int,
        simulator_out_dim: int,
        optimizer_config: OptimizerConfig,
        net_config: ModelConfig,
        *args,
        **kwargs
    ):
        optimizer_config = deepcopy(optimizer_config)
        net_config = deepcopy(net_config)

        super().__init__(*args, **kwargs)
        self.save_hyperparameters()

        self.net = SBINetwork(
            theta_dim=theta_dim,
            simulator_out_dim=simulator_out_dim,
            **net_config.__dict__,
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

    def training_step(self, batch: Tuple[Tensor, Tensor, Tensor], batch_idx: int):
        theta, simulator_out, x_target = batch
        network_res = self.forward(theta, x_target)
        loss = self.criterion.forward(network_res, simulator_out, x_target)
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

    def validation_step(self, batch: Tuple[Tensor, Tensor, Tensor], batch_idx: int):
        theta, simulator_out, x_target = batch
        network_res = self.forward(theta, x_target)
        loss = self.criterion.forward(network_res, simulator_out, x_target)
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

    def forward(self, theta: Tensor, x_target: Tensor) -> Tensor:
        """_summary_

        Args:
            prior (Tensor): (batch_size, n_prior_features)
            x_target (Tensor): (batch_size, n_target, n_sim_features)

        Returns:
            Tensor: (batch_size, n_target)
        """
        return self.net.forward(theta, x_target)

    def configure_optimizers(self):
        optimizer_cls = getattr(optim, self._optimizer_config.pop("name"))
        optimizer = optimizer_cls(self.parameters(), **self._optimizer_config)
        return optimizer


class Guidance(PotentialFunction):
    """time dependent version of the potential function
    """
    def __init__(
        self,
        theta_dim: int,
        simulator_out_dim: int,
        optimizer_config: OptimizerConfig,
        net_config: ModelConfig,
        diff_config: DiffusionConfig,
        *args,
        **kwargs
    ):
        modified_theta_dim = theta_dim
        if diff_config.include_t:
            # TODO: make this dependent on the diffusion time encoding
            modified_theta_dim += 1
        
        self.save_hyperparameters()
        super().__init__(
            modified_theta_dim,
            simulator_out_dim,
            optimizer_config,
            net_config,
            *args,
            **kwargs,
        )
        # from config2class.utils import deconstruct_config
        # print(deconstruct_config(diff_config))
        self.diff_schedule = self._build_schedule(deepcopy(diff_config))
        self.sampler = self._build_sampler(deepcopy(diff_config))
        self.diffusion_steps = diff_config.steps
        self.include_t = diff_config.include_t
        self.t = torch.linspace(0, 1, self.diffusion_steps)

    def _build_schedule(
        self, diff_config: DiffusionConfig
    ) -> diffusion_schedule.Schedule:
        schedule_name = diff_config.__dict__.pop("diffusion_schedule")
        schedule_cls = getattr(diffusion_schedule, schedule_name)
        schedule_config = getattr(diff_config, schedule_name)
        schedule_config.beta_schedule_cls = getattr(
            diffusion_schedule, schedule_config.beta_schedule_cls
        )
        schedule = schedule_cls(**schedule_config.__dict__)
        return schedule

    def _build_sampler(
        self, diff_config: DiffusionConfig
    ) -> diffusion_sampler.DiffSampler:
        sampler_name = diff_config.__dict__.pop("diffusion_time_sampler")
        sampler_cls = getattr(diffusion_sampler, sampler_name)
        sampler_config = getattr(diff_config, sampler_name)
        sampler = sampler_cls(**sampler_config.__dict__)
        return sampler

    def _get_diff_time_enc(self, t: Tensor) -> Tensor:
        """_summary_

        Args:
            t (Tensor): _description_

        Returns:
            Tensor: _description_
        """
        return t

    def _sample_diffusions(self, theta: Tensor, include_t: bool = False) -> Tensor:
        """sample a subset of thetas according to `self.sampler` and diffuse the samples
        according to `self.schedule`

        Args:
            theta (Tensor): (batch_size, theta_dim)
            include_t: (optional, bool): Would you like to append the time encoding into the sampled

        Returns:
            Tensor: (batch_size, sampled_diffs, theta_dim)
        """
        batch_size, theta_dim = theta.shape

        # sample a subset of diffusion time
        if self.training:
            sampled_t = self.sampler.forward_unbatched(self.t)
            sampled_t, _ = torch.sort(sampled_t)
        else:
            # eval model on bigger subsample
            # TODO: if you have more ram at disposal do this on whole batch
            sampled_t = self.t
            sampled_t = torch.linspace(0, 1, 100)  # this is hardcoded

        res = torch.empty((batch_size, len(sampled_t), theta_dim))
        insert_slice = slice(0, None)
        if sampled_t[0] == 0:
            # include the undiffused sample
            # sampled_t = sampled_t[1:]
            res[:, 0] = theta
            insert_slice = slice(1, None)
        diffused_theta = torch.normal(
            *self.diff_schedule.forward(theta, sampled_t[insert_slice])
        )
        res[:, insert_slice] = diffused_theta

        # add diff time. 
        if include_t:
            # TODO: write a get_time_enc function for more complex time-encodings
            batch_size = len(theta)
            sampled_t = sampled_t[None, :, None].repeat((batch_size, 1, 1))
            t_encoding = self._get_diff_time_enc(sampled_t)
            res = torch.cat([res, t_encoding], dim=-1)

        return res

    def training_step(self, batch, batch_idx):
        self.train()
        theta, simulator_out, x_target = batch

        theta_t = self._sample_diffusions(theta, include_t=self.include_t)
        network_res = self.forward(theta_t, x_target)
        loss = self.criterion.forward(network_res, simulator_out, x_target)
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

    def validation_step(self, batch, batch_idx):

        theta, simulator_out, x_target = batch

        theta_t = self._sample_diffusions(theta, include_t=self.include_t)
        network_res = self.forward(theta_t, x_target)
        loss = self.criterion.forward(network_res, simulator_out, x_target)
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

        tb_logger: TensorBoardLogger = self.loggers[0]

        fig, ax = plot_diffusion_step_loss(pred, d[:, 0], agg=True)
        tb_logger.experiment.add_figure(
            "val/diff_loss_plot", fig, global_step=self.global_step
        )
        fig, ax = plot_diffusion_step_corr(pred, d[:, 0], agg=True)
        tb_logger.experiment.add_figure(
            "val/diff_corr_plot", fig, global_step=self.global_step
        )

        plt.close(fig)

        self._val_step_outputs = {"pred": [], "d": []}
    
