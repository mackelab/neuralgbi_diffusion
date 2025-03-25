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
        target_noise_level: float = 0.01,
        n_target: int = 100,
        seed: int = 42,
        diffusion_scale: float = 0.5,
        max_diffusion_steps: int = 1000,
        n_misspecified: int = None,
        n_noised: int = 100,
        *args,
        **kwargs,
    ):
        super().__init__()

        self._theta: Tensor
        """simulator parameter"""

        self._x: Tensor
        """(specified) simulator result dependent on self._theta"""

        self._x_noised: Tensor
        """simulator results and noised with iid noised simulator results"""

        self._x_miss: Tensor
        """(misspecified) simulator result dependent on self._theta"""

        self._x_target: Tensor
        """concat of self._x, self._x_noised, self._x_mis"""

        self._target_noise_level = target_noise_level
        self._n_target = n_target  # how big should be the target subsample -> TODO: same as batch size
        self._seed = seed
        self._diffusion_scale = diffusion_scale
        self._max_diffusion_steps = max_diffusion_steps
        self._n_misspecified = n_misspecified
        self._n_noised = n_noised

    @classmethod
    def from_file(cls, path: str) -> "_SBIDataset":
        content: Dict[str, Tensor] = torch.load(
            path, weights_only=False, map_location=torch.device("cpu")
        )
        obj = cls(**content)
        for key, value in content.items():
            setattr(obj, key, value)
        setattr(obj, "_x_noised", obj._generate_x_noised())
        setattr(obj, "_x_all", obj._get_x_target())

        return obj

    def save(self, path: str):
        data = {
            "_theta": self._theta,
            "_x": self._x,
            "_x_mis:": self._x_miss,
            "_target_noise_std": self._target_noise_level,
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

    def generate_dataset(self, size: int):
        theta, x = self._sample_data(size)
        self._theta = theta
        self._x = x
        self._x_noised = self._generate_x_noised()
        self._x_miss = self._generate_misspecified_data()
        self._x_target = self._get_x_target()

    def _generate_misspecified_data(self) -> Tensor:
        n_misspecified = len(self._theta) if self._n_misspecified is None else self._n_misspecified
        x = self._x[: n_misspecified].clone()
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
        sample_idx = np.random.choice(
            len(self._x_target), size=self._n_target, replace=False
        )
        target_sample = self._x_target[sample_idx]
        return self._theta[index], self._x[index], target_sample

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

    def _generate_x_noised(self):
        random_state = np.random.default_rng(self._seed)
        
        n_samples = min(self._n_noised, len(self._x))
        x_sample = self._x[
            np.random.choice(len(self._x), size=n_samples, replace=False)
        ].clone()

        noise = random_state.standard_normal(size=x_sample.shape)
        res = x_sample + self._target_noise_level * self._x.std() * noise
        return res.float()

    def _get_x_target(self):
        concat = torch.cat([self._x, self._x_noised, self._x_miss], dim=0)
        return concat.float()


class _SourcererDataset(_SBIDataset):
    def __init__(
        self,
        target_noise_std=0.01,
        n_target=100,
        seed=42,
        diffusion_scale=0.5,
        max_diffusion_steps=1000,
        n_misspecified=None,
        n_noised=100,
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
            *args,
            **kwargs,
        )

    def _sample_data(self, size):
        cls_name = type(self).__name__ + "Simulator"
        simulator_cls = getattr(simulators, cls_name)
        simulator = simulator_cls()
        theta = simulator.sample_prior(size)
        x = simulator.sample(theta)
        return theta, x


class TwoMoons(_SourcererDataset):
    def __init__(
        self,
        target_noise_std=0.01,
        n_target=100,
        seed=42,
        diffusion_scale=0.5,
        max_diffusion_steps=1000,
        n_misspecified=None,
        n_noised=100,
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
        n_misspecified=None,
        n_noised=100,
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
        n_misspecified=None,
        n_noised=100,
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
        n_misspecified=None,
        n_noised=100,
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
        n_misspecified=None,
        n_noised=100,
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
            *args,
            **kwargs,
        )
        self._simulator = GaussianMixtureSimulator(seed=self._seed)

    def _sample_data(self, size):
        theta = self._simulator.prior.sample((size,))
        x = self._simulator.simulate(theta)
        # remove the trial dimension
        return theta, x

    def _generate_misspecified_data(self):
        if self._n_misspecified is None:
            n_samples = len(self._x)
        else:
            n_samples = min(self._n_misspecified, len(self._x))
        sample_idx = np.random.choice(len(self._x), size=n_samples, replace=False)
        x_miss = self._simulator.simulate_misspecified(self._theta[sample_idx])
        return x_miss


class LinearGaussian(_SBIDataset):
    def __init__(
        self,
        target_noise_std=0.01,
        n_target=100,
        seed=42,
        diffusion_scale=0.5,
        max_diffusion_steps=1000,
        n_misspecified=None,
        n_noised=1100,
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
            *args,
            **kwargs,
        )

        self._simulator = LinearGaussianSimulator(dim, seed=self._seed)

    def _sample_data(self, size):
        theta = self._simulator.prior.sample((size,))
        x = self._simulator.simulate(theta)
        return theta, x


class Uniform(_SBIDataset):
    def __init__(
        self,
        target_noise_std=0.01,
        n_target=100,
        seed=42,
        diffusion_scale=0.5,
        max_diffusion_steps=1000,
        n_misspecified=None,
        n_noised=1100,
        prior_bounds: Tuple = (-1.5, 1.5),
        poly_coeffs: Tensor = Tensor([0.1627, 0.9073, -1.2197, -1.4639, 1.4381]),
        epsilon: Union[Tensor, float] = 0.25,
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
            *args,
            **kwargs,
        )

        self._simulator = UniformNoise1DSimulator(
            prior_bounds=prior_bounds,
            seed=self._seed,
            poly_coeffs=poly_coeffs,
            epsilon=epsilon,
        )

    def _sample_data(self, size):
        theta = self._simulator.prior.sample((size,))
        x = self._simulator.simulate(theta)
        return theta, x
