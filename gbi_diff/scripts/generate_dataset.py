from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm
from gbi_diff.dataset import _SBIDataset  # pylint: disable=C0415
import gbi_diff.dataset.dataset as datasets
from gbi_diff.utils.cast import to_camel_case


def generate_dataset(
    dataset_type: str,
    sizes: List[int],
    path: str | Path = "./data",
    dataset_kwargs: Dict[str, Any] = None,
):
    if isinstance(path, str):
        path = Path(path)

    for size in tqdm(
        sizes, desc=f"Create datasets: {dataset_type}", position=0, leave=True
    ):
        cls_name = to_camel_case(dataset_type)
        cls_name = cls_name[0].upper() + cls_name[1:]
        dataset_cls = getattr(datasets, cls_name)
        dataset_kwargs = {} if dataset_kwargs is None else dataset_kwargs
        dataset: _SBIDataset = dataset_cls(**dataset_kwargs)
        dataset.generate_dataset(size)

        save_path = path / f"{dataset_type}_{size}.pt"
        dataset.save(save_path)
