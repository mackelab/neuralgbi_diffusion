from typing import Any, Dict
import yaml

def write_yaml(content: Dict[str, Any], path: str): 
    with open(path, "w", encoding="utf-8") as file:
        yaml.dump(content, file)
        