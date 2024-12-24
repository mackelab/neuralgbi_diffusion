from dataclasses import dataclass
from types import NoneType
import config2class.utils.filesystem as fs_utils


from config2class.utils import deconstruct_config


from config2class.utils.replacement import replace_tokens


@dataclass
class _SymmetricUniform:
    amplitude: int
    n_dims: int

    @classmethod
    def from_file(cls, file: str) -> "_SymmetricUniform":
        ending = file.split(".")[-1]
        content = getattr(fs_utils, f"load_{ending}")(file)
        content = replace_tokens(content)
        first_key, first_value = content.popitem()
        if len(content) == 0 and isinstance(first_value, dict):
            return cls(**first_value)
        else:
            content[first_key] = first_value
        return cls(**content)

    def to_file(self, file: str):
        ending = file.split(".")[-1]
        write_func = getattr(fs_utils, f"write_{ending}")
        content = deconstruct_config(self)
        write_func(file, content)


@dataclass
class Config:
    kernel: str
    beta: int
    prior: str
    warmup_steps: int
    SymmetricUniform: _SymmetricUniform

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

    def to_file(self, file: str):
        ending = file.split(".")[-1]
        write_func = getattr(fs_utils, f"write_{ending}")
        content = deconstruct_config(self)
        write_func(file, content)

    def __post_init__(self):
        self.SymmetricUniform = _SymmetricUniform(
            **self.SymmetricUniform
        )  # pylint: disable=E1134
