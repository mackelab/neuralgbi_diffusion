from typing import Dict
import torch
from gbi_diff.model.lit_module import SBI
from torch.distributions import Distribution


class PotentialFunc:
    def __init__(
        self, checkpoint: str, prior: Distribution, x_o: torch.Tensor, beta: float = 1
    ):
        """_summary_

        Args:
            check_point (str): path model checkpoint
            prior (Distribution): prior distribution
            x_o (torch.Tensor): observed data
            beta (float): inverse temperature
        """
        self.nn = SBI.load_from_checkpoint(checkpoint)
        self.prior = prior
        self.beta = beta

        if len(x_o.shape) == 1:
            # nest x_o if it is not a batch
            x_o = x_o[None]
        self.x_o = x_o

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

    def __call__(self, args: Dict[str, torch.Tensor]):
        theta = args["theta"]
        return self.potential_energy(theta)
