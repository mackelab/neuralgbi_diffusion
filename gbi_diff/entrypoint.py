from typing import List


class Process:
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

    def mcmc_sample(
        self,
        checkpoint: str,
        observed_data: str,
        size: int = 100,
        config_file: str = "config/mcmc.yaml",
        output: str = None,
        plot: bool = False,
    ):
        """sample mcmc

        Args:
            checkpoint (str): path to checkpoint
            observed_data (str): path to observed data.
            size (int, optional): how many samples you would like to sample. Defaults to 100.
            config_file (str, optional): path to config file. Defaults to "config/mcmc.yaml".
            output (str, optional): Directory where to store the sampled results. If this is None it will be a subdirectory in the checkpoint directory. Defaults to None
            plot (bool, optional): would like to create a pair-plot with your sampled data. Defaults to False
        """
        # >>>> add import here for faster help message
        import torch  # pylint: disable=C0415
        from gbi_diff.scripts.sampling import (
            sample_posterior,
            load_observed_data,
            save_samples
        )  # pylint: disable=C0415
        from gbi_diff.utils.mcmc_config import Config  # pylint: disable=C0415

        # <<<<

        config = Config.from_file(config_file)
        x_o = load_observed_data(observed_data)
        samples = sample_posterior(checkpoint, x_o, config, size)

        # save output
        save_samples(samples, checkpoint, output)

        if plot:
            print("plot results not implemented yet")
