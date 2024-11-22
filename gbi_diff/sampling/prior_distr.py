# Lotka Volterra
import torch
from torch.distributions import Distribution, MultivariateNormal, LogNormal, Uniform


class LotkaVolterraPrior(Distribution):
    def __init__(self, device: str = "cpu"):
        super().__init__(torch.Size(), torch.Size(), None)
        self.device = device

    def sample(self, size: int = 1):
        samples = torch.exp(
            torch.sigmoid(0.5 * torch.randn((size, 4), device=self.device))
        )
        samples = samples.squeeze()
        return samples

    def log_prob(self, value):
        raise NotImplementedError


# Inverse Kinematics
class IKPrior(Distribution):
    def __init__(self, var: float = None, device: str = "cpu"):
        super().__init__(torch.Size(), torch.Size(), None)
        if var is None:
            var = [1 / 16, 1 / 4, 1 / 4, 1 / 4]
        self.var = torch.tensor(var)
        self.device = device

    def sample(self, size: int = 1):
        prior = MultivariateNormal(
            torch.zeros(size, 4, device=self.device),
            self.var * torch.eye(4, device=self.device),
        )
        samples = prior.sample()
        samples = samples.squeeze()
        return samples

    def log_prob(self, value):
        raise NotImplementedError


# SIR
class SIRPrior(Distribution):
    def __init__(
        self,
    ):
        super().__init__(torch.Size(), torch.Size(), None)

    def sample(self, size: int = 1):
        beta_log = LogNormal(
            torch.log(torch.tensor([0.4], device=self.device)),
            torch.tensor([0.5], device=self.device),
        )
        gamma_log = LogNormal(
            torch.log(torch.tensor([0.125], device=self.device)),
            torch.tensor([0.2], device=self.device),
        )
        samples = torch.cat(
            [beta_log.sample((size,)), gamma_log.sample((size,))], dim=1
        )
        samples = samples.squeeze()
        return samples

    def log_prob(self, value):
        raise NotImplementedError


class SymmetricUniform(Uniform):
    def __init__(self, n_dims: int, amplitude: float = 1):
        ones = torch.ones(n_dims)
        super().__init__(-amplitude * ones, amplitude * ones, None)
