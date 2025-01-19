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
        """Suggested beta: [2.0, 10.0, 50.0]"""
        # Set seed.
        _ = torch.manual_seed(seed)
        self.prior = BoxUniform(-10 * ones(dim), 10 * ones(dim))
        self.num_trials = num_trials
        
    def simulate(self, theta: Tensor) -> Tensor:
        """Simulator."""
        samples1 = torch.randn((self.num_trials, *theta.shape)) + theta
        samples2 = 0.1 * torch.randn((self.num_trials, *theta.shape)) + theta
        all_samples = torch.zeros(*samples1.shape)

        bern = torch.bernoulli(0.5 * ones((self.num_trials, theta.shape[0]))).bool()

        all_samples[bern] = samples1[bern]
        all_samples[~bern] = samples2[~bern]
        all_samples = torch.permute(all_samples, (1, 0, 2))
        return all_samples

    def simulate_misspecified(self, theta: Tensor) -> Tensor:
        """Simulator."""
        # For misspecified x, push it out of the prior bounds.
        samples1 = torch.randn((self.num_trials, *theta.shape)) + theta
        samples2 = 0.5 * torch.randn((self.num_trials, *theta.shape)) + torch.sign(theta)*12.5
        all_samples = torch.zeros(*samples1.shape)

        bern = torch.bernoulli(0.5 * ones((self.num_trials, theta.shape[0]))).bool()

        all_samples[bern] = samples1[bern]
        all_samples[~bern] = samples2[~bern]
        all_samples = torch.permute(all_samples, (1, 0, 2))
        # NOTE: this line is functional in the original implementation but does not work in our case:
        # assert ((all_samples[:,:,0]>self.limits[0,0]) & (all_samples[:,:,0]<self.limits[0,1]) & (all_samples[:,:,1]>self.limits[1,0]) & (all_samples[:,:,1]<self.limits[1,1])).all()
        return all_samples