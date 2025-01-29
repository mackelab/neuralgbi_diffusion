from abc import ABC, abstractmethod
from pathlib import Path

from omegaconf import DictConfig
from torch import Tensor

from gbi_diff.sampling.utils import save_torch


class PosteriorSampler(ABC):
    def __init__(self):
        super().__init__()
        self._config: DictConfig
        self.x_o: Tensor

    @abstractmethod
    def forward(self, n_sample: int) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def single_forward(self, x_o: Tensor, n_samples: int) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def _get_default_path(self) -> Path:
        """gets the default save path where to save the produced samples

        Returns:    
            Path: _description_
        """
        raise NotImplementedError

    def save_samples(
        self,
        x_o: Tensor,
        samples: Tensor,
        output: str = None,
        file_name: str = "samples.pt",
    ):
        """save samples to file system

        Args:
            x_o (Tensor): _description_
            samples (Tensor): sampled data
            output (Path, optional): output directory. If none is given it will the checkpoint directory. Defaults to None.
            file_name (str, optional): How to name the file. Has to end with ".pt". Defaults to "samples.pt".
        """
        if output is None:
            output = self._get_default_path()
        elif isinstance(output, str):
            output = Path(output)
        output = output / file_name

        config = self._config
        if not isinstance(config, dict):
            config = config.to_container()
        data = {"x_o": x_o, "theta": samples, **config}

        print(f"Save samples at: {output}")
        save_torch(data, output)
