from .early_stopping import EarlyStopping
from .checkpoint import Checkpoint

def build_early_stopping(es_cfg):
    """Create EarlyStopping, or None if disabled."""
    if not es_cfg.get('enabled', False):
        return None
    return EarlyStopping(patience=es_cfg.patience, delta=es_cfg.delta, verbose=es_cfg.verbose)

def build_checkpoint(ckpt_cfg, run_dir: str):
    """Create Checkpoint, or None if disabled."""
    if not ckpt_cfg.get('enabled', False):
        return None
    return Checkpoint(save_dir=run_dir)
