from dataclasses import dataclass
from types import NoneType
from config2class.api.base import StructuredConfig


@dataclass
class _SymmetricUniform(StructuredConfig):
    amplitude: int = None
    n_dims: int = None


@dataclass
class Config(StructuredConfig):
    kernel: str = None
    beta: int = None
    observed_data_file: str = None
    prior: str = None
    warmup_steps: int = None
    SymmetricUniform: _SymmetricUniform = None

    def __post_init__(self):
        self.SymmetricUniform = _SymmetricUniform(
            **self.SymmetricUniform
        )  # pylint: disable=E1134
