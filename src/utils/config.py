from omegaconf import OmegaConf
from src import CONFIG_DIR

def get_config(name: str = "train_config.yaml"):
    cfg = OmegaConf.load(CONFIG_DIR / name)
    # Add computed values
    cfg.run_dir = f"{cfg.experiment.save_dir}/{cfg.experiment.save_name}"
    return cfg
