from typing import List
from omegaconf import DictConfig
from pyargwriter.decorator import add_hydra


class Entrypoint:
    """CLI Process to handle GBI pipeline"""

    def __init__(self):
        """init an instance of Process"""

    def generate_data(
        self,
        dataset_type: str,
        sizes: List[int],
        path: str = "./data",
    ):
        """creates a specified dataset and stores it into the file system.

        Args:
            dataset_type (str): dataset_type for dataset: currently available: moon
            sizes (int): how many samples you want to create
            path (str): directory where you want to store the dataset
        """
        # >>>> add import here for faster help message
        from gbi_diff.scripts.generate_dataset import (
            generate_dataset,
        )  # pylint: disable=C0415

        # <<<<

        generate_dataset(dataset_type, sizes, path)

    def train_potential(
        self,
        config_file: str = "config/train_potential.yaml",
        device: int = 1,
        force: bool = False,
    ):
        """start training process as defined in your config file

        Args:
            config_file (str): path to config file (allowed are yaml, toml and json). Defaults to: "config/train_potential.yaml"
            device (int, optional): set to a number to indicate multiple devices. Defaults to 1.
            force (bool, optional): If you would like to start training without any questions
        """
        # >>>> add import here for faster help message
        from gbi_diff.scripts.train import train_potential  # pylint: disable=C0415
        from gbi_diff.utils.train_potential_config import Config  # pylint: disable=C0415

        # <<<<

        config = Config.from_file(config_file, resolve=True)
        train_potential(config, device, force)

    def train_guidance(
        self,
        config_file: str = "config/train_guidance.yaml",
        device: int = 1,
        force: bool = False,
    ):
        """start training process as defined in your config file

        Args:
            config_file (str): path to config file (allowed are yaml, toml and json). Defaults to: "config/train_guidance.yaml"
            device (int, optional): set to a number to indicate multiple devices. Defaults to 1.
            force (bool, optional): If you would like to start training without any questions
        """
        # >>>> add import here for faster help message
        from gbi_diff.scripts.train import train_guidance  # pylint: disable=C0415
        from gbi_diff.utils.train_guidance_config import (
            Config,
        )  # pylint: disable=C0415

        # <<<<

        config = Config.from_file(config_file, resolve=True)
        train_guidance(config, device, force)

    def train_diffusion(
        self,
        config_file: str = "config/train_diffusion.yaml",
        device: int = 1,
        force: bool = False,
    ):
        """train diffusion model which is also the prior for the sampling process

        Args:
            config_file (str, optional): _description_. Defaults to "config/train_diffusion.yaml".
            device (int, optional): _description_. Defaults to 1.
            force (bool, optional): _description_. Defaults to False.
        """
        # >>>> add import here for faster help message
        from gbi_diff.scripts.train import train_diffusion  # pylint: disable=C0415
        from gbi_diff.utils.train_diffusion_config import (
            Config,
        )  # pylint: disable=C0415

        # <<<<
        config = Config.from_file(config_file, resolve=True)
        train_diffusion(config, device, force)

    @add_hydra(
        "config",
        version_base=None,
        config_name="sampling_diffusion.yaml",
        config_path="config/",
    )
    def diffusion_sample(
        self,
        config: DictConfig,
        diffusion_ckpt: str,
        guidance_ckpt: str,
        output: str = None,
        n_samples: int = 100,
        plot: bool = False,
    ):
        """sample from diffusion process

        Args:
            diffusion_ckpt (str): path to checkpoint of diffusion model
            guidance_ckpt (str): path to checkpoint of guidance model
            config (str): path to config to use for diffusion sample. Defaults to "config/sampling_diffusion.yaml".
            output (str, optional): Where to store the samples. If None: store alongside diffusion process. Defaults to None.
            n_samples (int, optional): How many samples to sample from posterior distribution. Defaults to 100.
            plot (bool, optional): Would you like to add pair plots of the posterior distribution. Defaults to False.
        """
        # >>>> add import here for faster help message
        from gbi_diff.scripts.sampling import (
            diffusion_sampling,
        )  # pylint: disable=C0415
        from gbi_diff.utils.sampling_diffusion_config import (
            Config,
        )  # pylint: disable=C0415

        # <<<<

        config: Config = Config.from_dict_config(config)

        diffusion_sampling(
            diffusion_ckpt, guidance_ckpt, config, output, n_samples, plot
        )

    def mcmc_sample(
        self,
        checkpoint: str,
        n_samples: int = 100,
        config_file: str = "config/sampling_mcmc.yaml",
        output: str = None,
        plot: bool = False,
        num_worker: int = 1,
    ):
        """sample mcmc

        Args:
            checkpoint (str): path to checkpoint
            n_samples (int, optional): how many samples you would like to sample. Defaults to 100.
            config_file (str, optional): path to config file. Defaults to "config/mcmc.yaml".
            output (str, optional): Directory where to store the sampled results. If this is None it will be a subdirectory in the checkpoint directory. Defaults to None
            plot (bool, optional): would like to create a pair-plot with your sampled data. Defaults to False
            num_worker (int): How many threads you would like to use to sample from mcmc
        """
        if num_worker > 1:
            raise NotImplementedError(
                "Multithread for parallel sampling is not done yet. "
            )

        # >>>> add import here for faster help message
        from gbi_diff.utils.sampling_mcmc_config import Config  # pylint: disable=C0415
        from gbi_diff.scripts.sampling import mcmc_sampling  # pylint: disable=C0415

        # <<<<
        config = Config.from_file(config_file)
        mcmc_sampling(checkpoint, config, output, n_samples, plot)

    @add_hydra(
        "config",
        version_base=None,
        config_name="sampling_diffusion.yaml",
        config_path="config/",
    )
    def evaluate_diffusion_sampling(
        self,
        config: DictConfig,
        diffusion_ckpt: str,
        guidance_ckpt: str,
        output: str = None,
        n_samples: int = 100,
        plot: bool = False,
    ):
        pass
