from dataclasses import dataclass
from types import NoneType
from config2class.api.base import StructuredConfig


@dataclass
class _Dataset(StructuredConfig):
    train_file: str
    val_file: str
    n_target: int
    noise_level: int


@dataclass
class _TimeEncoder(StructuredConfig):
    input_dim: int
    output_dim: int
    activation_func: str
    architecture: list
    final_activation: str


@dataclass
class _ThetaEncoder(StructuredConfig):
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
    TimeEncoder: _TimeEncoder
    ThetaEncoder: _ThetaEncoder
    LatentMLP: _LatentMLP

    def __post_init__(self):
        self.TimeEncoder = _TimeEncoder(**self.TimeEncoder)  #pylint: disable=E1134
        self.ThetaEncoder = _ThetaEncoder(**self.ThetaEncoder)  #pylint: disable=E1134
        self.LatentMLP = _LatentMLP(**self.LatentMLP)  #pylint: disable=E1134


@dataclass
class _VPSchedule(StructuredConfig):
    beta_start: float
    beta_end: float
    T: str
    beta_schedule_cls: str


@dataclass
class _DDPMSchedule(StructuredConfig):
    beta_start: float
    beta_end: float
    T: str
    beta_schedule_cls: str


@dataclass
class _Diffusion(StructuredConfig):
    steps: int
    time_repr_dim: str
    period_spread: int
    diffusion_schedule: str
    VPSchedule: _VPSchedule
    DDPMSchedule: _DDPMSchedule

    def __post_init__(self):
        self.VPSchedule = _VPSchedule(**self.VPSchedule)  #pylint: disable=E1134
        self.DDPMSchedule = _DDPMSchedule(**self.DDPMSchedule)  #pylint: disable=E1134


@dataclass
class _Optimizer(StructuredConfig):
    name: str
    lr: float
    weight_decay: float


@dataclass
class Config(StructuredConfig):
    data_entity: str
    results_dir: str
    check_val_every_n_epochs: int
    num_worker: int
    max_epochs: int
    batch_size: int
    precision: int
    dataset: _Dataset
    model: _Model
    diffusion: _Diffusion
    optimizer: _Optimizer

    def __post_init__(self):
        self.dataset = _Dataset(**self.dataset)  #pylint: disable=E1134
        self.model = _Model(**self.model)  #pylint: disable=E1134
        self.diffusion = _Diffusion(**self.diffusion)  #pylint: disable=E1134
        self.optimizer = _Optimizer(**self.optimizer)  #pylint: disable=E1134
