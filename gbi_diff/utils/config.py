from dataclasses import dataclass
import config2class.utils.filesystem as fs_utils


from config2class.utils.replacement import replace_tokens


@dataclass
class _Model:
    latent_dim: int
    theta_encoder: list
    simulator_encoder: list
    latent_mlp: list
    activation_func: str

    @classmethod
    def from_file(cls, file: str) -> "_Model":
        ending = file.split(".")[-1]
        content = getattr(fs_utils, f"load_{ending}")(file)
        content = replace_tokens(content)
        first_key, first_value = content.popitem()
        if len(content) == 0 and isinstance(first_value, dict):
            return cls(**first_value)
        else:
            content[first_key] = first_value
        return cls(**content)


@dataclass
class _Optimizer:
    name: str
    lr: float
    weight_decay: int

    @classmethod
    def from_file(cls, file: str) -> "_Optimizer":
        ending = file.split(".")[-1]
        content = getattr(fs_utils, f"load_{ending}")(file)
        content = replace_tokens(content)
        first_key, first_value = content.popitem()
        if len(content) == 0 and isinstance(first_value, dict):
            return cls(**first_value)
        else:
            content[first_key] = first_value
        return cls(**content)


@dataclass
class Config:
    results_dir: str
    check_val_every_n_epochs: int
    num_worker: int
    train_file: str
    val_file: str
    n_target: int
    max_epochs: int
    batch_size: int
    precision: int
    model: _Model
    optimizer: _Optimizer

    @classmethod
    def from_file(cls, file: str) -> "Config":
        ending = file.split(".")[-1]
        content = getattr(fs_utils, f"load_{ending}")(file)
        content = replace_tokens(content)
        first_key, first_value = content.popitem()
        if len(content) == 0 and isinstance(first_value, dict):
            return cls(**first_value)
        else:
            content[first_key] = first_value
        return cls(**content)

    def __post_init__(self):
        self.model = _Model(**self.model)  # pylint: disable=E1134
        self.optimizer = _Optimizer(**self.optimizer)  # pylint: disable=E1134
