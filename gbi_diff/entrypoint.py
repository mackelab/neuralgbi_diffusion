from typing import List

from tqdm import tqdm


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
        from gbi_diff.dataset import SBIDataset  # pylint: disable=C0415

        # <<<<
        for size in tqdm(sizes, desc="Create datasets"):
            dataset = SBIDataset()
            dataset.generate_dataset(size, dataset_type)
            dataset.save(path.rstrip("/") + f"/{dataset_type}_{size}.pt")

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
        from gbi_diff.experiment import train  # pylint: disable=C0415
        from gbi_diff.utils.config import Config  # pylint: disable=C0415

        # <<<<
        config = Config.from_file(config_file)
        train(config, device, force)
