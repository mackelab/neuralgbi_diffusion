from typing import List

from tqdm import tqdm
from gbi_diff.dataset import SBIDataset  # pylint: disable=C0415


def generate_dataset(dataset_type: str, sizes: List[int], path: str = "./data"):
    for size in tqdm(sizes, desc="Create datasets"):
        dataset = SBIDataset()
        dataset.generate_dataset(size, dataset_type)
        dataset.save(path.rstrip("/") + f"/{dataset_type}_{size}.pt")
