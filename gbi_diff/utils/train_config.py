from dataclasses import dataclass
from types import NoneType
import config2class.utils.filesystem as fs_utils


from config2class.utils import deconstruct_config


from config2class.utils.replacement import replace_tokens


@dataclass
class _Dataset:
    train_file: str
    val_file: str
    n_target: int
    noise_std: float

    @classmethod
    def from_file(cls, file: str) -> "_Dataset":
        ending = file.split('.')[-1]
        content = getattr(fs_utils, f'load_{ending}')(file)
        content = replace_tokens(content)
        first_key, first_value = content.popitem()
        if len(content) == 0 and isinstance(first_value, dict):
            return cls(**first_value)
        else:
            content[first_key] = first_value
        return cls(**content)

    def to_file(self, file: str):
        ending = file.split('.')[-1]
        write_func = getattr(fs_utils, f'write_{ending}')
        content = deconstruct_config(self)
        write_func(file, content)


@dataclass
class _Model:
    latent_dim: int
    theta_encoder: list
    simulator_encoder: list
    latent_mlp: list
    activation_func: str
    final_activation: NoneType

    @classmethod
    def from_file(cls, file: str) -> "_Model":
        ending = file.split('.')[-1]
        content = getattr(fs_utils, f'load_{ending}')(file)
        content = replace_tokens(content)
        first_key, first_value = content.popitem()
        if len(content) == 0 and isinstance(first_value, dict):
            return cls(**first_value)
        else:
            content[first_key] = first_value
        return cls(**content)

    def to_file(self, file: str):
        ending = file.split('.')[-1]
        write_func = getattr(fs_utils, f'write_{ending}')
        content = deconstruct_config(self)
        write_func(file, content)


@dataclass
class _UniformSampler:
    p: int

    @classmethod
    def from_file(cls, file: str) -> "_UniformSampler":
        ending = file.split('.')[-1]
        content = getattr(fs_utils, f'load_{ending}')(file)
        content = replace_tokens(content)
        first_key, first_value = content.popitem()
        if len(content) == 0 and isinstance(first_value, dict):
            return cls(**first_value)
        else:
            content[first_key] = first_value
        return cls(**content)

    def to_file(self, file: str):
        ending = file.split('.')[-1]
        write_func = getattr(fs_utils, f'write_{ending}')
        content = deconstruct_config(self)
        write_func(file, content)


@dataclass
class _VPSchedule:
    beta_min: float
    beta_max: float
    beta_schedule_cls: str

    @classmethod
    def from_file(cls, file: str) -> "_VPSchedule":
        ending = file.split('.')[-1]
        content = getattr(fs_utils, f'load_{ending}')(file)
        content = replace_tokens(content)
        first_key, first_value = content.popitem()
        if len(content) == 0 and isinstance(first_value, dict):
            return cls(**first_value)
        else:
            content[first_key] = first_value
        return cls(**content)

    def to_file(self, file: str):
        ending = file.split('.')[-1]
        write_func = getattr(fs_utils, f'write_{ending}')
        content = deconstruct_config(self)
        write_func(file, content)


@dataclass
class _Diffusion:
    enabled: bool
    steps: int
    diffusion_time_sampler: str
    diffusion_schedule: str
    UniformSampler: _UniformSampler
    VPSchedule: _VPSchedule

    @classmethod
    def from_file(cls, file: str) -> "_Diffusion":
        ending = file.split('.')[-1]
        content = getattr(fs_utils, f'load_{ending}')(file)
        content = replace_tokens(content)
        first_key, first_value = content.popitem()
        if len(content) == 0 and isinstance(first_value, dict):
            return cls(**first_value)
        else:
            content[first_key] = first_value
        return cls(**content)

    def to_file(self, file: str):
        ending = file.split('.')[-1]
        write_func = getattr(fs_utils, f'write_{ending}')
        content = deconstruct_config(self)
        write_func(file, content)

    def __post_init__(self):
        self.UniformSampler = _UniformSampler(**self.UniformSampler)  #pylint: disable=E1134
        self.VPSchedule = _VPSchedule(**self.VPSchedule)  #pylint: disable=E1134


@dataclass
class _Optimizer:
    name: str
    lr: float
    weight_decay: int

    @classmethod
    def from_file(cls, file: str) -> "_Optimizer":
        ending = file.split('.')[-1]
        content = getattr(fs_utils, f'load_{ending}')(file)
        content = replace_tokens(content)
        first_key, first_value = content.popitem()
        if len(content) == 0 and isinstance(first_value, dict):
            return cls(**first_value)
        else:
            content[first_key] = first_value
        return cls(**content)

    def to_file(self, file: str):
        ending = file.split('.')[-1]
        write_func = getattr(fs_utils, f'write_{ending}')
        content = deconstruct_config(self)
        write_func(file, content)


@dataclass
class Config:
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

    @classmethod
    def from_file(cls, file: str) -> "Config":
        ending = file.split('.')[-1]
        content = getattr(fs_utils, f'load_{ending}')(file)
        content = replace_tokens(content)
        first_key, first_value = content.popitem()
        if len(content) == 0 and isinstance(first_value, dict):
            return cls(**first_value)
        else:
            content[first_key] = first_value
        return cls(**content)

    def to_file(self, file: str):
        ending = file.split('.')[-1]
        write_func = getattr(fs_utils, f'write_{ending}')
        content = deconstruct_config(self)
        write_func(file, content)

    def __post_init__(self):
        self.dataset = _Dataset(**self.dataset)  #pylint: disable=E1134
        self.model = _Model(**self.model)  #pylint: disable=E1134
        self.diffusion = _Diffusion(**self.diffusion)  #pylint: disable=E1134
        self.optimizer = _Optimizer(**self.optimizer)  #pylint: disable=E1134
