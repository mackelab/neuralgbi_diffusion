# This file was copied from: https://github.com/mackelab/neuralgbi/blob/main/gbi/benchmark/tasks/gaussian_mixture/task.py
# Additional slight modifications where applied

from torch import ones, Tensor
import torch
from sbi.utils import BoxUniform


class GaussianMixtureSimulator:
    def __init__(
        self,
        num_trials: int = 5,
        dim: int = 2,
        seed: int = 0,
    ):
        """Init Gaussian Mixture Simulator

        Args:
            num_trials (int, optional): simulate how many samples per theta. Defaults to 5.
            dim (int, optional): dimension of gaussian mixture. Defaults to 2.
            seed (int, optional): random seed. Defaults to 0.
        """
        # Set seed.
        _ = torch.manual_seed(seed)
        self.prior = BoxUniform(-10 * ones(dim), 10 * ones(dim))
        self.num_trials = num_trials

    def simulate(self, theta: Tensor) -> Tensor:
        """simulate gaussian clusters based on given params theta.
        For each given parameter combination in theta sample n_trials many samples for a gaussian blob

        Args:
            theta (Tensor): base parameters (n_samples, theta_dim)

        Returns:
            Tensor: simulated gaussian clusters (n_samples, n_trials, sim_out_dim)
        """
        samples1 = torch.randn((self.num_trials, *theta.shape)) + theta
        samples2 = 0.1 * torch.randn((self.num_trials, *theta.shape)) + theta
        all_samples = torch.zeros(*samples1.shape)

        bern = torch.bernoulli(0.5 * ones((self.num_trials, theta.shape[0]))).bool()

        all_samples[bern] = samples1[bern]
        all_samples[~bern] = samples2[~bern]
        all_samples = torch.permute(all_samples, (1, 0, 2))
        return all_samples

    def simulate_misspecified(self, theta: Tensor) -> Tensor:
        """simulate gaussian clusters based on given params theta.
        But also misspecify a portion of the given samples so the relation theta -> x is not necessarily given
        For each given parameter combination in theta sample n_trials many samples for a gaussian blob

        Args:
            theta (Tensor): base parameters (n_samples, theta_dim)

        Returns:
            Tensor: simulated gaussian clusters (n_samples, n_trials, sim_out_dim)
        """
        # For misspecified x, push it out of the prior bounds.
        samples1 = torch.randn((self.num_trials, *theta.shape)) + theta
        samples2 = (
            0.5 * torch.randn((self.num_trials, *theta.shape))
            + torch.sign(theta) * 12.5
        )
        all_samples = torch.zeros(*samples1.shape)

        bern = torch.bernoulli(0.5 * ones((self.num_trials, theta.shape[0]))).bool()

        all_samples[bern] = samples1[bern]
        all_samples[~bern] = samples2[~bern]
        all_samples = torch.permute(all_samples, (1, 0, 2))
        # NOTE: this line is functional in the original implementation but does not work in our case:
        # assert ((all_samples[:,:,0]>self.limits[0,0]) & (all_samples[:,:,0]<self.limits[0,1]) & (all_samples[:,:,1]>self.limits[1,0]) & (all_samples[:,:,1]<self.limits[1,1])).all()
        return all_samples
