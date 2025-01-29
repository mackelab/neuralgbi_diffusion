from pathlib import Path
from typing import Dict

import numpy as np
import torch
from pyro import infer
from pyro.infer import MCMC
from torch.distributions import Distribution
from tqdm import tqdm

from gbi_diff.model.lit_module import PotentialNetwork
from gbi_diff.sampling import prior_distr
from gbi_diff.sampling.sampler import PosteriorSampler
from gbi_diff.sampling.utils import get_sample_path, load_observed_data
from gbi_diff.utils.plot import _pair_plot
from gbi_diff.utils.sampling_mcmc_config import Config


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
        self.nn = PotentialNetwork.load_from_checkpoint(checkpoint)
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
        if self.nn._net._theta_encoder._input_dim != prior_dim:
            raise ValueError(
                f"Theta dim from prior does not fit to theta dim from likelihood:{self.nn.net._theta_encoder._input_dim} != {prior_dim}"
            )
        return True


class MCMCSampler(PosteriorSampler):
    def __init__(self, checkpoint: str | Path, config: Config):
        super().__init__()
        self._checkpoint = checkpoint
        self._config = config
        self._x_o, _ = load_observed_data(self._config.observed_data_file)

        self._potential_function = self._create_potential_fn(checkpoint)

    def _create_potential_fn(self, checkpoint: str) -> PotentialFunc:
        prior_config = getattr(self._config, self._config.prior)
        prior_cls = getattr(prior_distr, self._config.prior)
        prior = prior_cls(**prior_config.to_container())

        potential_func = PotentialFunc(
            checkpoint=checkpoint, prior=prior, x_o=self._x_o, beta=self._config.beta
        )
        potential_func.is_valid()
        return potential_func

    def _get_default_path(self):
        return get_sample_path(self._checkpoint)
    
    def single_forward(self, x_o: torch.Tensor, n_samples: int) -> torch.Tensor:
        """single forward of one observed data point

        Args:
            x_o (torch.Tensor): (feature_dim, )
            n_samples (int): how many samples you would like to create

        Returns:
            torch.Tensor: (n_samples, theta_dim)
        """
        self._potential_function.update_x_o(x_o)
        kernel_cls = getattr(infer, self._config.kernel)
        kernel = kernel_cls(potential_fn=self._potential_function)

        mcmc = MCMC(
            kernel,
            num_samples=n_samples,
            warmup_steps=self._config.warmup_steps,
            initial_params={"theta": self._potential_function.prior.sample()},
        )
        mcmc.run(x_o)
        samples = mcmc.get_samples()
        return samples["theta"]

    def forward(self, n_samples: int) -> torch.Tensor:
        """_summary_

        Args:
            n_samples (int): _description_

        Returns:
            torch.Tensor: (n_samples, num_observed_data, theta_dim)
        """
        batch_size = len(self._x_o)
        res = torch.zeros((n_samples, batch_size, self.theta_dim))
        for idx, x in enumerate(tqdm(self._x_o, desc="Sample in observed data")):
            res[:, idx] = self.single_forward(x, n_samples)
        return res

    def pair_plot(
        self, samples: torch.Tensor, x_o: torch.Tensor, output: str | Path = None
    ):
        """_summary_

        Args:
            samples (torch.Tensor): (n_samples, n_target, theta_dim) n_target: how many x_o, n_samples: samples per x_o
            x_o (torch.Tensor): (n_target, sim_out_dim)
            output (str | Path, optional): where to store the images. If None: store them alongside the checkpoint
        """
        batch_size, n_target, _ = samples.shape
        x_o = x_o[None].repeat(batch_size, 1, 1)

        if output is None:
            save_dir = self._get_default_path()
        elif isinstance(output, Path):
            save_dir = output
        else:
            save_dir = Path(output)
            
        for target_idx in range(n_target):
            self._potential_function.update_x_o(x_o[target_idx])
            log_prob = self._potential_function.log_likelihood(samples[:, target_idx])
            sample = samples[:, target_idx]
            title = f"Index: {target_idx}, beta: {self._config.beta}"
            file_name = f"pair_plot_{target_idx}_beta_{self._config.beta}.png"
            save_path = save_dir / file_name
            _pair_plot(
                sample, torch.exp(log_prob), title=title, save_path=str(save_path)
            )

    def update_beta(self, value: float):
        assert isinstance(value, (float, int)), f"Expected numeric, got {type(value)}"
        self._config.beta = value
        self._potential_function.beta = value

    @property
    def theta_dim(self) -> int:
        return self._potential_function.nn.hparams.theta_dim

    @property
    def potential_function(self) -> PotentialFunc:
        return self._potential_function
