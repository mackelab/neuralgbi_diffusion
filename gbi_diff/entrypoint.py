from typing import List


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

    def train(
        self,
        config_file: str = "config/train.yaml",
        device: int = 1,
        force: bool = False,
    ):
        """start training process as defined in your config file

        Args:
            config_file (str): path to config file (allowed are yaml, toml and json). Defaults to: "config/train.yaml"
            device (int, optional): set to a number to indicate multiple devices. Defaults to 1.
            force (bool, optional): If you would like to start training without any questions
        """
        # >>>> add import here for faster help message
        from gbi_diff.scripts.train import train  # pylint: disable=C0415
        from gbi_diff.utils.train_config import Config  # pylint: disable=C0415

        # <<<<

        config = Config.from_file(config_file)
        train(config, device, force)

    def train_guidance(
        self,
        config_file: str = "config/train_theta_noise.yaml",
        device: int = 1,
        force: bool = False,
    ):
        """start training process as defined in your config file

        Args:
            config_file (str): path to config file (allowed are yaml, toml and json). Defaults to: "config/train.yaml"
            device (int, optional): set to a number to indicate multiple devices. Defaults to 1.
            force (bool, optional): If you would like to start training without any questions
        """
        # >>>> add import here for faster help message
        from gbi_diff.scripts.train import train_guidance  # pylint: disable=C0415
        from gbi_diff.utils.train_guidance_config import (
            Config,
        )  # pylint: disable=C0415

        # <<<<

        config = Config.from_file(config_file)
        train_guidance(config, device, force)

    def mcmc_sample(
        self,
        checkpoint: str,
        observed_data: str,
        size: int = 100,
        config_file: str = "config/mcmc.yaml",
        output: str = None,
        plot: bool = False,
        num_worker: int = 1,
    ):
        """sample mcmc

        Args:
            checkpoint (str): path to checkpoint
            observed_data (str): path to observed data.
            size (int, optional): how many samples you would like to sample. Defaults to 100.
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
        import torch  # pylint: disable=C0415

        from gbi_diff.sampling.utils import (  # pylint: disable=C0415
            load_observed_data,
            save_samples,
        )
        from gbi_diff.scripts.sampling import (  # pylint: disable=C0415
            sample_posterior,
        )
        from gbi_diff.utils.mcmc_config import Config  # pylint: disable=C0415
        from gbi_diff.utils.plot import pair_plot_stack  # pylint: disable=C0415

        # <<<<

        config = Config.from_file(config_file)
        x_o = load_observed_data(observed_data)
        samples = sample_posterior(checkpoint, x_o, config, size)

        # save output
        save_samples(samples, checkpoint, output)

        if plot:
            pair_plot_stack(samples, checkpoint, config, output)
