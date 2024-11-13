from typing import Tuple

from matplotlib import pyplot as plt
import torch
from lightning import LightningModule
from lightning.pytorch.loggers import TensorBoardLogger
from torch import Tensor, optim

from gbi_diff.model.networks import SBINetwork
from gbi_diff.utils.config import _Model as ModelConfig
from gbi_diff.utils.config import _Optimizer as OptimizerConfig
from gbi_diff.utils.criterion import SBICriterion
from gbi_diff.utils.plot import plot_correlation


class SBI(LightningModule):
    def __init__(
        self,
        prior_dim: int,
        simulator_out_dim: int,
        optimizer_config: OptimizerConfig,
        net_config: ModelConfig,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()

        
        self.net = SBINetwork(
            theta_dim=prior_dim,
            simulator_out_dim=simulator_out_dim,
            **net_config.__dict__
        )

        self.example_input_array = (
            torch.zeros(1, prior_dim),
            torch.zeros(1, 1, simulator_out_dim),
        )

        self.criterion = SBICriterion(distance_order=2)
        self._optimizer_config = optimizer_config.__dict__

        # this thing should not leave the class. Inconsistencies with strings feared
        self._train_step_outputs = {"pred": [], "d": []}
        self._val_step_outputs = {"pred": [], "d": []}


    def training_step(self, batch: Tuple[Tensor, Tensor, Tensor], batch_idx: int):
        prior, simulator_out, x_target = batch
        network_res = self.forward(prior, x_target)
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
        prior, simulator_out, x_target = batch
        network_res = self.forward(prior, x_target)
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
        fig, _ = plot_correlation(pred, d)

        tb_logger: TensorBoardLogger = self.loggers[0]
        tb_logger.experiment.add_figure(
            "val/corr_plot", fig, global_step=self.global_step
        )

        plt.close(fig)
        self._val_step_outputs = {"pred": [], "d": []}

    def forward(self, prior: Tensor, x_target: Tensor) -> Tensor:
        """_summary_

        Args:
            prior (Tensor): (batch_size, n_prior_features)
            x_target (Tensor): (batch_size, n_target, n_sim_features)

        Returns:
            Tensor: (batch_size, n_target)
        """
        return self.net.forward(prior, x_target)

    def configure_optimizers(self):
        optimizer_cls = getattr(optim, self._optimizer_config.pop("name"))
        optimizer = optimizer_cls(self.parameters(), **self._optimizer_config)
        return optimizer
