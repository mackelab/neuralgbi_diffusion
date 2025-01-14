from dataclasses import dataclass
from types import NoneType
from config2class.api.base import StructuredConfig


@dataclass
class Config(StructuredConfig):
    beta: int
    observed_data_file: str
