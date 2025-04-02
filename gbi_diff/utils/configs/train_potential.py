from dataclasses import dataclass
from types import NoneType
from config2class.api.base import StructuredConfig


@dataclass
class _Dataset(StructuredConfig):
    train_file: str = None
    val_file: str = None
    n_target: int = None
    noise_level: int = None


@dataclass
class _TimeEncoder(StructuredConfig):
    enabled: bool = None
    input_dim: int = None
    n_layers: int = None
    hidden_dim: int = None


@dataclass
class _ThetaEncoder(StructuredConfig):
    enabled: bool = None
    output_dim: int = None
    n_layers: int = None
    hidden_dim: int = None


@dataclass
class _SimulatorEncoder(StructuredConfig):
    enabled: bool = None
    n_layers: int = None
    hidden_dim: int = None
    output_dim: int = None


@dataclass
class _LatentMLP(StructuredConfig):
    net_type: str = None
    n_target: str = None
    n_layers: int = None
    hidden_dim: int = None
    dropout_prob: float = None
    use_batch_norm: bool = None


@dataclass
class _Model(StructuredConfig):
    TimeEncoder: _TimeEncoder = None
    ThetaEncoder: _ThetaEncoder = None
    SimulatorEncoder: _SimulatorEncoder = None
    LatentMLP: _LatentMLP = None
    final_activation: str = None
    standardize: bool = None

    def __post_init__(self):
        self.TimeEncoder = _TimeEncoder(**self.TimeEncoder)  # pylint: disable=E1134
        self.ThetaEncoder = _ThetaEncoder(**self.ThetaEncoder)  # pylint: disable=E1134
        self.SimulatorEncoder = _SimulatorEncoder(
            **self.SimulatorEncoder
        )  # pylint: disable=E1134
        self.LatentMLP = _LatentMLP(**self.LatentMLP)  # pylint: disable=E1134


@dataclass
class _Optimizer(StructuredConfig):
    name: str = None
    lr: float = None
    weight_decay: float = None


@dataclass
class Config(StructuredConfig):
    data_entity: str = None
    check_val_every_n_epochs: int = None
    num_worker: int = None
    max_epochs: int = None
    batch_size: int = None
    precision: int = None
    dataset: _Dataset = None
    model: _Model = None
    optimizer: _Optimizer = None

    def __post_init__(self):
        self.dataset = _Dataset(**self.dataset)  # pylint: disable=E1134
        self.model = _Model(**self.model)  # pylint: disable=E1134
        self.optimizer = _Optimizer(**self.optimizer)  # pylint: disable=E1134
