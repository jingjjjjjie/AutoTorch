'''
Loss function registry and builder.
Add new loss functions by adding them to LOSS_FN_MAP in registry.py.
'''
from .registry import LOSS_FN_MAP


def build_loss_fn(cfg):
    """Create loss function from config.

    Supported: cross_entropy, bce, bce_with_logits, mse
    """
    name = cfg['training']['loss_fn']
    if name not in LOSS_FN_MAP:
        raise ValueError(f"Unknown loss function '{name}'. Available: {list(LOSS_FN_MAP.keys())}")
    return LOSS_FN_MAP[name]()
