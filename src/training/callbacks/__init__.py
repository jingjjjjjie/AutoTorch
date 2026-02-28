from .early_stopping import EarlyStopping
from .checkpoint import Checkpoint

def build_early_stopping(cfg):
    """Create EarlyStopping from config, or None if disabled."""
    es = cfg.get('early_stopping', {})
    if not es.get('enabled', False):
        return None
    return EarlyStopping(patience=es['patience'], delta=es['delta'], verbose=es['verbose'])

def build_checkpoint(cfg):
    """Create Checkpoint from config, or None if disabled."""
    ckpt = cfg.get('checkpoint', {})
    if not ckpt.get('enabled', False):
        return None
    return Checkpoint(
        save_dir=cfg['model']['save_dir'],
        save_name=cfg['model']['save_name'],
    )
