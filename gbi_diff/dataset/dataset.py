from pathlib import Path
from typing import Dict, Tuple, Union
import torch
from torch import Tensor
from torch.utils.data import Dataset
import numpy as np

from abc import abstractmethod
from sourcerer import simulators

from gbi_diff.dataset.simulators.gaussian_mixture import GaussianMixtureSimulator
from gbi_diff.dataset.simulators.linear_gaussian import LinearGaussianSimulator
from gbi_diff.dataset.simulators.uniform import UniformNoise1DSimulator
from gbi_diff.dataset.utils import generate_x_misspecified


class _SBIDataset(Dataset):
    def __init__(
        self,
        target_noise_std: float = 0.01,
        n_target: int = 100,
        seed: int = 42,
        diffusion_scale: float = 0.5,
        max_diffusion_steps: int = 1000,
        n_misspecified: int = 20,
        n_noised: int = 100,
        normalize: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__()

        self._theta: Tensor
        """simulator parameter"""

        self._x: Tensor
        """simulator result dependent on self._theta"""

        self._x_target: Tensor
        """simulator results and noised with iid noised simulator results"""

        self._x_miss: Tensor
        """misspecified data"""

        self._x_all: Tensor
        """concat of self._x, self._target, self._measured"""

        self._target_noise_std = target_noise_std
        self._n_target = n_target  # how big should be the target subsample -> TODO: same as batch size
        self._seed = seed
        self._diffusion_scale = diffusion_scale
        self._max_diffusion_steps = max_diffusion_steps
        self._n_misspecified = n_misspecified
        self._n_noised = n_noised
        self._normalize = bool(normalize)  # compensate also for None

        self._theta_stats: Tuple[Tensor, Tensor]  # mean and std
        self._x_stats: Tuple[Tensor, Tensor]  # mean and std

    @classmethod
    def from_file(cls, path: str, **kwargs) -> "_SBIDataset":
        content: Dict[str, Tensor] = torch.load(
            path, weights_only=False, map_location=torch.device("cpu")
        )
        # overwrite arguments from file if given
        content = {**content, **kwargs}
        obj = cls(**content)
        for key, value in content.items():
            setattr(obj, key, value)
        setattr(obj, "_x_target", obj._generate_x_target())
        setattr(obj, "_x_miss", obj._generate_misspecified_data())
        setattr(obj, "_x_all", obj._get_all())

        obj._theta_stats, obj._x_stats = obj.get_stats()

        return obj

    def save(self, path: str):
        data = {
            "_theta": self._theta,
            "_x": self._x,
            "_target_noise_std": self._target_noise_std,
            "_seed": self._seed,
            "_diffusion_scale": self._diffusion_scale,
            "_max_diffusion_steps": self._max_diffusion_steps,
            "_n_misspecified": self._n_misspecified,
            "_n_noised": self._n_noised,
        }
        torch.save(data, path)

    @abstractmethod
    def _sample_data(self, size: int) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError

    @abstractmethod
    def sample_posterior(self, prior_samples: Tensor) -> Tensor:
        raise NotImplementedError

    def generate_dataset(self, size: int):
        theta, x = self._sample_data(size)
        self._theta = theta
        self._x = x
        self._x_target = self._generate_x_target()
        self._x_miss = self._generate_misspecified_data()
        self._x_all = self._get_all()

        self._theta_stats, self._x_stats = self.get_stats()

    def _generate_misspecified_data(self) -> Tensor:
        x = self._x[: self._n_misspecified].clone()
        x_miss = generate_x_misspecified(
            x, self._diffusion_scale, self._max_diffusion_steps
        )
        return x_miss

    def get_theta_dim(self) -> int:
        return self._theta.shape[1]

    def get_sim_out_dim(self) -> int:
        return self._x.shape[1]

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Tensor]:
        """_summary_

        Args:
            index (int): _description_

        Returns:
            Tuple[Tensor, Tensor, Tensor]: theta, x, x_target
        """
        theta = self._theta[index]
        x = self._x[index]

        sample_idx = np.random.choice(
            len(self._x_all), size=self._n_target, replace=False
        )
        target_sample = self._x_all[sample_idx]

        if self._normalize:
            theta = self.normalize_theta(theta)
            x = self.normalize_x(x)
            target_sample = self.normalize_x(target_sample)
        return theta, x, target_sample

    def __len__(self) -> int:
        """return length of prior (theta)

        Returns:
            int: length of theta
        """
        try:
            return len(self._theta)
        except AttributeError:
            return 0

    def set_n_target(self, value: int):
        self._n_target = value

    def _generate_x_target(self):
        random_state = np.random.default_rng(self._seed)
        n_samples = min(self._n_noised, len(self._x))
        x_sample = self._x[
            np.random.choice(len(self._x), size=n_samples, replace=False)
        ].clone()
        noise = random_state.normal(0, 1, size=x_sample.shape)
        res = x_sample + noise * self._target_noise_std
        return res.float()

    def _get_all(self):
        concat = torch.cat([self._x, self._x_target, self._x_miss], dim=0)
        return concat.float()

    def get_theta_mean(self) -> Tensor:
        return self._theta.mean()

    def get_theta_std(self) -> Tensor:
        return self._theta.std()

    def get_x_mean(self) -> Tensor:
        return self._x.mean()

    def get_x_std(self) -> Tensor:
        return self._x.std()

    def get_stats(self) -> Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]:
        """_summary_

        Returns:
            Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]: mean and std for theta and x
        """
        return (
            (self.get_theta_mean(), self.get_theta_std()),
            (self.get_x_mean(), self.get_x_std()),
        )

    def set_stats(self, stats: Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]):
        self._theta_stats, self._x_stats = stats
    
    def save_stats(self, path: Path | str = None):
        """save stats in data_stats.pt

        Args:
            path (Path | str): path to directory, save as data_stats.pt
        """
        theta_mean, theta_std = self._theta_stats
        x_mean, x_std = self._x_stats

        if path is None:
            path = Path.cwd()
        elif isinstance(path, str):
            path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save({"theta_mean": theta_mean, "theta_std": theta_std, "x_mean": x_mean, "x_std": x_std}, path.joinpath("data_stats.pt"))

    def normalize_theta(self, theta: Tensor) -> Tensor:
        theta_mean, theta_std = self._theta_stats
        return (theta - theta_mean) / theta_std

    def unnormalize_theta(self, theta: Tensor) -> Tensor:
        theta_mean, theta_std = self._theta_stats
        return theta * theta_std + theta_mean

    def normalize_x(self, x: Tensor) -> Tensor:
        x_mean, x_std = self._x_stats
        return (x - x_mean) / x_std

    def unnormalize_x(self, x: Tensor) -> Tensor:
        x_mean, x_std = self._x_stats
        return x * x_std + x_mean

    def __repr__(self):
        s = f"{type(self).__name__}:\n\t{self._target_noise_std=}\n\t{self._n_target=}\n\t{self._seed=}\n\t{self._diffusion_scale=}\n\t{self._max_diffusion_steps=}\n\t{self._n_misspecified=}\n\t{self._n_noised=}\n\t{self._normalize=}\n\tself._theta_stats=({self.get_theta_mean()},{self.get_theta_std()})\n\tself._x_stats=({self.get_x_mean()},{self.get_x_std()})"
        return s


class _SourcererDataset(_SBIDataset):
    def __init__(
        self,
        target_noise_std=0.01,
        n_target=100,
        seed=42,
        diffusion_scale=0.5,
        max_diffusion_steps=1000,
        n_misspecified=20,
        n_noised=100,
        normalize=False,
        *args,
        **kwargs,
    ):
        super().__init__(
            target_noise_std,
            n_target,
            seed,
            diffusion_scale,
            max_diffusion_steps,
            n_misspecified,
            n_noised,
            normalize,
            *args,
            **kwargs,
        )
        cls_name = type(self).__name__ + "Simulator"
        simulator_cls = getattr(simulators, cls_name)
        self.simulator = simulator_cls()

    def sample_posterior(self, prior_samples):
        x = self.simulator.sample(prior_samples)
        return x

    def _sample_data(self, size):
        theta = self.simulator.sample_prior(size)
        x = self.sample_posterior(theta)
        return theta, x


class TwoMoons(_SourcererDataset):
    def __init__(
        self,
        target_noise_std=0.01,
        n_target=100,
        seed=42,
        diffusion_scale=0.5,
        max_diffusion_steps=1000,
        n_misspecified=20,
        n_noised=100,
        normalize=False,
        *args,
        **kwargs,
    ):
        super().__init__(
            target_noise_std,
            n_target,
            seed,
            diffusion_scale,
            max_diffusion_steps,
            n_misspecified,
            n_noised,
            normalize,
            *args,
            **kwargs,
        )


class LotkaVolterra(_SourcererDataset):
    def __init__(
        self,
        target_noise_std=0.01,
        n_target=100,
        seed=42,
        diffusion_scale=0.5,
        max_diffusion_steps=1000,
        n_misspecified=20,
        n_noised=100,
        normalize=False,
        *args,
        **kwargs,
    ):
        super().__init__(
            target_noise_std,
            n_target,
            seed,
            diffusion_scale,
            max_diffusion_steps,
            n_misspecified,
            n_noised,
            normalize,
            *args,
            **kwargs,
        )


class InverseKinematics(_SourcererDataset):
    def __init__(
        self,
        target_noise_std=0.01,
        n_target=100,
        seed=42,
        diffusion_scale=0.5,
        max_diffusion_steps=1000,
        n_misspecified=20,
        n_noised=100,
        normalize=False,
        *args,
        **kwargs,
    ):
        super().__init__(
            target_noise_std,
            n_target,
            seed,
            diffusion_scale,
            max_diffusion_steps,
            n_misspecified,
            n_noised,
            normalize,
            *args,
            **kwargs,
        )


class SIR(_SourcererDataset):
    def __init__(
        self,
        target_noise_std=0.01,
        n_target=100,
        seed=42,
        diffusion_scale=0.5,
        max_diffusion_steps=1000,
        n_misspecified=20,
        n_noised=100,
        normalize=False,
        *args,
        **kwargs,
    ):
        super().__init__(
            target_noise_std,
            n_target,
            seed,
            diffusion_scale,
            max_diffusion_steps,
            n_misspecified,
            n_noised,
            normalize,
            *args,
            **kwargs,
        )


class GaussianMixture(_SBIDataset):
    def __init__(
        self,
        target_noise_std=0.01,
        n_target=100,
        seed=42,
        diffusion_scale=0.5,
        max_diffusion_steps=1000,
        n_misspecified=20,
        n_noised=1100,
        normalize=False,
        *args,
        **kwargs,
    ):
        super().__init__(
            target_noise_std,
            n_target,
            seed,
            diffusion_scale,
            max_diffusion_steps,
            n_misspecified,
            n_noised,
            normalize,
            *args,
            **kwargs,
        )
        self.simulator = GaussianMixtureSimulator(num_trials=1, seed=self._seed)

    def sample_posterior(self, prior_samples):
        x = self.simulator.simulate(prior_samples)
        # remove the trial dimension
        x = x[:, 0]
        return x

    def _sample_data(self, size):
        theta = self.simulator.prior.sample((size,))
        x = self.sample_posterior(theta)
        return theta, x

    def _generate_misspecified_data(self):
        n_samples = min(self._n_misspecified, len(self._x))
        sample_idx = np.random.choice(len(self._x), size=n_samples, replace=False)
        x_miss = self.simulator.simulate_misspecified(self._theta[sample_idx])
        # remove the trial dimension
        x_miss = x_miss[:, 0]
        return x_miss


class LinearGaussian(_SBIDataset):
    def __init__(
        self,
        target_noise_std=0.01,
        n_target=100,
        seed=42,
        diffusion_scale=0.5,
        max_diffusion_steps=1000,
        n_misspecified=20,
        n_noised=1100,
        normalize=False,
        dim: int = 10,
        *args,
        **kwargs,
    ):
        super().__init__(
            target_noise_std,
            n_target,
            seed,
            diffusion_scale,
            max_diffusion_steps,
            n_misspecified,
            n_noised,
            normalize,
            *args,
            **kwargs,
        )

        self._simulator = LinearGaussianSimulator(dim, seed=self._seed)

    def sample_posterior(self, prior_samples):
        return self._simulator.simulate(prior_samples)

    def _sample_data(self, size):
        theta = self._simulator.prior.sample((size,))
        x = self.sample_posterior(theta)
        return theta, x


class Uniform(_SBIDataset):
    def __init__(
        self,
        target_noise_std=0.01,
        n_target=100,
        seed=42,
        diffusion_scale=0.5,
        max_diffusion_steps=1000,
        n_misspecified=20,
        n_noised=1100,
        prior_bounds: Tuple = (-1.5, 1.5),
        poly_coeffs: Tensor = Tensor([0.1627, 0.9073, -1.2197, -1.4639, 1.4381]),
        epsilon: Union[Tensor, float] = 0.25,
        normalize=False,
        *args,
        **kwargs,
    ):
        super().__init__(
            target_noise_std,
            n_target,
            seed,
            diffusion_scale,
            max_diffusion_steps,
            n_misspecified,
            n_noised,
            normalize,
            *args,
            **kwargs,
        )

        self.simulator = UniformNoise1DSimulator(
            prior_bounds=prior_bounds,
            seed=self._seed,
            poly_coeffs=poly_coeffs,
            epsilon=epsilon,
        )

    def sample_posterior(self, prior_samples):
        return self.simulator.simulate(prior_samples)

    def _sample_data(self, size):
        theta = self.simulator.prior.sample((size,))
        x = self.sample_posterior(theta)
        return theta, x
