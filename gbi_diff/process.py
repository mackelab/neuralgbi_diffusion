from gbi_diff.utils.config import Config


class Process:
    """"""

    def __init__(self):
        """init an instance of Process"""

    def generate_data(
        self,
        dataset_type: str,
        size: int,
        path: str = "./data",
        noise_std: float = 0.01,
    ):
        """creates a specified dataset and stores it into the file system.

        Args:
            dataset_type (str): dataset_type for dataset: currently available: moon
            size (int): how many samples you want to create
            path (str): directory where you want to store the dataset
            noise_std (float, optional): hwo to noise the data. Defaults to 0.01.
        """
        # >>>> add import here for faster help message
        from gbi_diff.dataset import SBIDataset

        # <<<<

        n_target = size
        dataset = SBIDataset(noise_std, None, n_target)
        dataset.generate_dataset(size, dataset_type)
        path = path.rstrip("/") + f"/{dataset_type}_{size}.pt"
        dataset.store(path)

    def train(self, config_file: str = "config/train.yaml"):
        """start training process as defined in your config file

        Args:
            config_file (str): path to config file (allowed are yaml, toml and json). Defaults to: "config/train.yaml"
        """
        # >>>> add import here for faster help message
        from gbi_diff.experiment import train

        # <<<<
        config = Config.from_file(config_file)
        train(config)