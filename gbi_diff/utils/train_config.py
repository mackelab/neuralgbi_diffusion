from dataclasses import dataclass
from types import NoneType
from config2class.api.base import StructuredConfig


@dataclass
class _Dataset(StructuredConfig):
    train_file: str
    val_file: str
    n_target: int
    noise_std: float


@dataclass
class _ThetaEncoder(StructuredConfig):
    output_dim: int
    architecture: list
    activation_func: str
    final_activation: str


@dataclass
class _SimulatorEncoder(StructuredConfig):
    output_dim: int
    architecture: list
    activation_func: str
    final_activation: str


@dataclass
class _LatentMLP(StructuredConfig):
    architecture: list
    activation_func: str
    final_activation: NoneType


@dataclass
class _Model(StructuredConfig):
    ThetaEncoder: _ThetaEncoder
    SimulatorEncoder: _SimulatorEncoder
    LatentMLP: _LatentMLP

    def __post_init__(self):
        self.ThetaEncoder = _ThetaEncoder(**self.ThetaEncoder)  # pylint: disable=E1134
        self.SimulatorEncoder = _SimulatorEncoder(
            **self.SimulatorEncoder
        )  # pylint: disable=E1134
        self.LatentMLP = _LatentMLP(**self.LatentMLP)  # pylint: disable=E1134


@dataclass
class _Optimizer(StructuredConfig):
    name: str
    lr: float
    weight_decay: int


@dataclass
class Config(StructuredConfig):
    results_dir: str
    check_val_every_n_epochs: int
    num_worker: int
    max_epochs: int
    batch_size: int
    precision: int
    dataset: _Dataset
    model: _Model
    optimizer: _Optimizer

    def __post_init__(self):
        self.dataset = _Dataset(**self.dataset)  # pylint: disable=E1134
        self.model = _Model(**self.model)  # pylint: disable=E1134
        self.optimizer = _Optimizer(**self.optimizer)  # pylint: disable=E1134
