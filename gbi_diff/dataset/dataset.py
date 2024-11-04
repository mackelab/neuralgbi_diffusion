from typing import Dict, Literal, Tuple
import torch
from torch import Tensor
from torch.utils.data import Dataset
import numpy as np

from gbi_diff.dataset.utils import generate_moon
import logging


class SBIDataset(Dataset):
    def __init__(
        self,
        target_noise_std: float = 0.01,
        measured: Tensor = None,
        n_target: int = 100,
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

    @classmethod
    def from_file(cls, path: str) -> "SBIDataset":
        content: Dict[str, Tensor] = torch.load(path, weights_only=False, map_location=torch.device("cpu"))
        print(content)
        obj = cls(**content)
        for key, value in content.items():
            setattr(obj, key, value)
        return obj

    def load_file(self, path: str):
        content: Dict[str, Tensor] = torch.load(path, map_location=torch.device("cpu"))
        for key, value in content.items():
            setattr(self, key, value)

    def store(self, path: str):
        data = {
            "_theta": self._theta,
            "_x": self._x,
            "_target": self._target,
            "_measured": self._measured,
            "_target_noise_std": self._target_noise_std,
        }
        torch.save(data, path)

    def generate_dataset(self, size: int, type: Literal["moon"] = "moon"):
        # TODO: add more datasets
        if type != "moon":
            msg = f"{type} is not available as type"
            logging.warning(msg)
        theta, x = generate_moon(size)
        self._theta = theta
        self._x = x
        self._target = self._x + torch.randn(self._x.size()) * self._target_noise_std

        self._all = torch.cat([self._x, self._target], dim=0)
        if self._measured is not None:
            self._all = torch.cat([self._all, self._measured], dim=0)

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
