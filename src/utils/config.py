from pathlib import Path
from omegaconf import OmegaConf
from src import CONFIG_DIR

def get_config(name: str = "train_config.yaml", path: str = None):
    config_path = Path(path) if path else CONFIG_DIR / name
    cfg = OmegaConf.load(config_path)
    cfg.run_dir = f"{cfg.experiment.save_dir}/{cfg.experiment.save_name}"
    return cfg
