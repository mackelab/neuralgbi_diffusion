from dataclasses import dataclass
from types import NoneType
from config2class.api.base import StructuredConfig


@dataclass
class Config(StructuredConfig):
    betas: list
    n_samples: int
    data_entity: str
    observed_data_file: str
    simulator: str
