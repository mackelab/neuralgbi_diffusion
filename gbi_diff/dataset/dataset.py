from typing import Dict, Literal, Tuple
import torch
from torch import Tensor
from torch.utils.data import Dataset
import numpy as np

from gbi_diff.dataset import simulator_api
import logging


class SBIDataset(Dataset):
    def __init__(
        self,
        target_noise_std: float = 0.01,
        measured: Tensor = None,
        n_target: int = 100,
        seed: int = 42,
        *args,
        **kwargs,
    ):
        super().__init__()

        self._theta: Tensor
        """simulator parameter"""
        self._x: Tensor
        """simulator result dependent on self._theta"""

        self._target: Tensor
        """simulator results and noised with iid noised simulator results"""

        self._measured = measured
        """data I actually measured"""

        self._all: Tensor
        """concat of self._x, self._target, self._measured"""

        self._target_noise_std = target_noise_std
        self._n_target = n_target  # how big should be the target subsample -> TODO: same as batch size
        self._seed = seed

    @classmethod
    def from_file(cls, path: str) -> "SBIDataset":
        content: Dict[str, Tensor] = torch.load(
            path, weights_only=False, map_location=torch.device("cpu")
        )
        obj = cls(**content)
        for key, value in content.items():
            setattr(obj, key, value)
        setattr(obj, "_target", obj._get_x_target())
        setattr(obj, "_all", obj._get_all())

        return obj

    def load_file(self, path: str):
        content: Dict[str, Tensor] = torch.load(path, map_location=torch.device("cpu"))
        for key, value in content.items():
            setattr(self, key, value)
        self._target = self._get_x_target()
        self._all = self._get_all()

    def save(self, path: str):
        data = {
            "_theta": self._theta,
            "_x": self._x,
            "_measured": self._measured,
            "_target_noise_std": self._target_noise_std,
            "_seed": self._seed,
        }
        torch.save(data, path)

    def generate_dataset(self, size: int, entity: str = "moon"):
        prefix = "generate_"
        generator_func_names = list(
            filter(lambda x: prefix in x, simulator_api.__dict__.keys())
        )
        types = list(map(lambda x: "_".join(x.split("_")[1:]), generator_func_names))
        if entity not in types:
            raise NotImplementedError(
                f"There is not simulator api implemented for the `entity`: {entity}\
                                      Available are: {types}"
            )
        func = getattr(simulator_api, prefix + entity)
        theta, x = func(size)
        self._theta = theta
        self._x = x
        self._target = self._get_x_target()
        self._all = self._get_all()

    def get_prior_dim(self) -> int:
        return self._theta.shape[1]

    def get_sim_out_dim(self) -> int:
        return self._x.shape[1]

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Tensor]:
        target_sample = self._all[
            np.random.choice(len(self._all), size=self._n_target, replace=False)
        ]
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

    def _get_x_target(self):
        random_state = np.random.default_rng(self._seed)
        noise = random_state.normal(0, self._target_noise_std, size=self._x.size())
        res = self._x + noise
        return res.float()

    def _get_all(self):
        concat = torch.cat([self._x, self._target], dim=0)
        if self._measured is not None:
            concat = torch.cat([concat, self._measured], dim=0)
        return concat.float()
