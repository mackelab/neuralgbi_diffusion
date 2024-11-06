from dataclasses import dataclass

from typing import Dict, Any
import yaml
import json
import toml


def _load_yaml(path: str, encoding: str = "utf-8") -> Dict[Any, Any]:
    with open(path, encoding=encoding) as file:
        content = yaml.safe_load(file)
    return content


def _load_json(path: str, encoding: str = "utf-8") -> Dict[Any, Any]:
    with open(path, encoding=encoding) as file:
        content = json.load(file)
    return content


def _load_toml(path: str, encoding: str = "utf-8") -> Dict[Any, Any]:
    with open(path, encoding=encoding) as file:
        content = toml.load(file)
    return content


@dataclass
class _Optimizer:
    name: str
    lr: float
    weight_decay: int

    @classmethod
    def from_file(cls, file: str) -> "_Optimizer":
        ending = file.split(".")[-1]
        content = globals()[f"_load_{ending}"](file)
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
    optimizer: _Optimizer

    @classmethod
    def from_file(cls, file: str) -> "Config":
        ending = file.split(".")[-1]
        content = globals()[f"_load_{ending}"](file)
        first_key, first_value = content.popitem()
        if len(content) == 0 and isinstance(first_value, dict):
            return cls(**first_value)
        else:
            content[first_key] = first_value
        return cls(**content)

    def __post_init__(self):
        self.optimizer = _Optimizer(**self.optimizer)  # pylint: disable=E1134
