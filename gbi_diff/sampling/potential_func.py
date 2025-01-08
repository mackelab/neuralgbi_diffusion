from typing import Dict
import torch
from gbi_diff.model.lit_module import PotentialFunction
from torch.distributions import Distribution


class PotentialFunc:
    def __init__(
        self,
        checkpoint: str,
        prior: Distribution,
        x_o: torch.Tensor = None,
        beta: float = 1,
    ):
        """_summary_

        Args:
            check_point (str): path model checkpoint
            prior (Distribution): prior distribution
            x_o (torch.Tensor): observed data
            beta (float): inverse temperature
        """
        self.nn = PotentialFunction.load_from_checkpoint(checkpoint)
        self.prior = prior
        self.beta = beta

        self.x_o = x_o
        if self.x_o is not None:
            self.update_x_o(self.x_o)

    def log_likelihood(self, theta: torch.Tensor) -> torch.Tensor:
        x_o = self.x_o
        batched = False
        if len(theta.shape) == 2:
            x_o = self.x_o[None].repeat(len(theta), 1, 1)
            batched = True
        score = self.nn.forward(theta, x_o)
        ll = -self.beta * score
        if batched:
            ll = ll[:, 0]
        return ll

    def log_posterior(self, theta: torch.Tensor) -> torch.Tensor:
        log_posterior = self.log_likelihood(theta) + self.log_prior(theta)
        return log_posterior

    def potential_energy(self, theta: torch.Tensor) -> torch.Tensor:
        log_posterior = self.log_posterior(theta)
        # negative log density
        return -log_posterior

    def log_prior(self, theta: torch.Tensor) -> torch.Tensor:
        # NOTE: assume the dimensions of the prior are iid
        dim = None
        if len(theta.shape) == 2:
            theta = theta[:, None]
            dim = -1
        log_prior = self.prior.log_prob(theta).sum(dim=dim)
        return log_prior

    def update_x_o(self, value: torch.Tensor):
        if len(value.shape) == 1:
            # nest value if it is not a batch
            value = value[None]
        self.x_o = value

    def __call__(self, args: Dict[str, torch.Tensor]):
        theta = args["theta"]
        return self.potential_energy(theta)

    def is_valid(self) -> bool:
        """check if the potential function is consistent

        Returns:
            bool: if the potential function will work as specified
        """
        prior_dim = len(self.prior.sample())
        if self.nn.net._theta_encoder._input_dim != prior_dim:
            raise ValueError(
                f"Theta dim from prior does not fit to theta dim from likelihood:{self.nn.net._theta_encoder._input_dim} != {prior_dim}"
            )
        return True
