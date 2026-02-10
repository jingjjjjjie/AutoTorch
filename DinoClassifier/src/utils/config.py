import yaml
from pathlib import Path

config_dir = Path(__file__).parent.parent / "configs"

def load_config(config_path: str) -> dict:
    """Load YAML config file and return as dictionary."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_config(config_name: str = "train_config.yaml", config_dir=config_dir) -> dict:
    """Load config from the configs directory."""
    return load_config(config_dir / config_name)
