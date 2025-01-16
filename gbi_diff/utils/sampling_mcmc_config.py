from dataclasses import dataclass
from types import NoneType
from config2class.api.base import StructuredConfig


@dataclass
class _SymmetricUniform(StructuredConfig):
    amplitude: int
    n_dims: int


@dataclass
class Config(StructuredConfig):
    kernel: str
    beta: int
    observed_data_file: str
    prior: str
    warmup_steps: int
    SymmetricUniform: _SymmetricUniform

    def __post_init__(self):
        self.SymmetricUniform = _SymmetricUniform(
            **self.SymmetricUniform
        )  # pylint: disable=E1134
