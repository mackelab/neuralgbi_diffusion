from typing import Dict, Tuple

import torch
from lightning import LightningModule
from torch import Tensor, optim

from gbi_diff.model.networks import SBINetwork
from gbi_diff.utils.criterion import SBICriterion


class SBI(LightningModule):
    def __init__(
        self,
        prior_dim: int,
        simulator_out_dim: int,
        optimizer_config: Dict[str, float | str],
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.net = SBINetwork(
            theta_dim=prior_dim, simulator_out_dim=simulator_out_dim, latent_dim=256
        )
        self.criterion = SBICriterion(distance_order=2)
        self._optimizer_config = optimizer_config

    def training_step(self, batch: Tuple[Tensor, Tensor, Tensor], batch_idx: int):
        prior, simulator_out, x_target = batch
        network_res = self.forward(prior, x_target)
        loss = self.criterion.forward(network_res, simulator_out, x_target)
        return loss

    def validation_step(self, batch: Tuple[Tensor, Tensor, Tensor], batch_idx: int):
        prior, simulator_out, x_target = batch
        network_res = self.forward(prior, x_target)
        loss = self.criterion.forward(network_res, simulator_out, x_target)
        return loss

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
        optimizer = optimizer_cls(
            self.parameters(), **self._optimizer_config
        )
        return optimizer
