'''
Optimizer and scheduler registry and builders.
Add new optimizers to OPTIMIZER_MAP in registry.py.
Add new schedulers in scheduler.py.
'''
from .registry import OPTIMIZER_MAP
from .scheduler import build_scheduler


def build_optimizer(cfg, model):
    """Create optimizer from config.

    Supported: adam, adamw, sgd
    """
    name = cfg['training']['optimizer']
    if name not in OPTIMIZER_MAP:
        raise ValueError(f"Unknown optimizer '{name}'. Available: {list(OPTIMIZER_MAP.keys())}")
    lr = float(cfg['training']['learning_rate'])
    weight_decay = float(cfg['training']['weight_decay'])
    return OPTIMIZER_MAP[name](model.parameters(), lr=lr, weight_decay=weight_decay)
