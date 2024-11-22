import os
from typing import Any, Dict
import yaml


def write_yaml(content: Dict[str, Any], path: str):
    directory = "/".join(path.split("/")[:-1])
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(path, "w", encoding="utf-8") as file:
        yaml.dump(content, file)
