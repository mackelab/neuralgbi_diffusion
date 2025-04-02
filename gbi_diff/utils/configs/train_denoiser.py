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
    input_dim: int = None
    output_dim: int = None
    activation_func: str = None
    architecture: list = None
    final_activation: str = None


@dataclass
class _ThetaEncoder(StructuredConfig):
    output_dim: int = None
    architecture: list = None
    activation_func: str = None
    final_activation: str = None


@dataclass
class _LatentMLP(StructuredConfig):
    architecture: list = None
    activation_func: str = None
    final_activation: NoneType = None


@dataclass
class _Model(StructuredConfig):
    TimeEncoder: _TimeEncoder = None
    ThetaEncoder: _ThetaEncoder = None
    LatentMLP: _LatentMLP = None

    def __post_init__(self):
        self.TimeEncoder = _TimeEncoder(**self.TimeEncoder)  # pylint: disable=E1134
        self.ThetaEncoder = _ThetaEncoder(**self.ThetaEncoder)  # pylint: disable=E1134
        self.LatentMLP = _LatentMLP(**self.LatentMLP)  # pylint: disable=E1134


@dataclass
class _VPSchedule(StructuredConfig):
    beta_start: float = None
    beta_end: float = None
    T: str = None
    beta_schedule_cls: str = None


@dataclass
class _DDPMSchedule(StructuredConfig):
    beta_start: float = None
    beta_end: float = None
    T: str = None
    beta_schedule_cls: str = None


@dataclass
class _Diffusion(StructuredConfig):
    steps: int = None
    time_repr_dim: str = None
    period_spread: int = None
    diffusion_schedule: str = None
    VPSchedule: _VPSchedule = None
    DDPMSchedule: _DDPMSchedule = None

    def __post_init__(self):
        self.VPSchedule = _VPSchedule(**self.VPSchedule)  # pylint: disable=E1134
        self.DDPMSchedule = _DDPMSchedule(**self.DDPMSchedule)  # pylint: disable=E1134


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
    diffusion: _Diffusion = None
    optimizer: _Optimizer = None

    def __post_init__(self):
        self.dataset = _Dataset(**self.dataset)  # pylint: disable=E1134
        self.model = _Model(**self.model)  # pylint: disable=E1134
        self.diffusion = _Diffusion(**self.diffusion)  # pylint: disable=E1134
        self.optimizer = _Optimizer(**self.optimizer)  # pylint: disable=E1134
