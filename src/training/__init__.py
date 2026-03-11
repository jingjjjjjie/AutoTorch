'''
Training components: loss, optimizers, callbacks, and trainers.
'''
from .loss import LOSS_FN_MAP
from .optimizers import OPTIMIZER_MAP
from .lr_scheduler import LRScheduler

def build_loss_fn(name: str):
    """Create loss function. Supported: cross_entropy, bce, bce_with_logits, mse"""
    if name not in LOSS_FN_MAP:
        raise ValueError(f"Unknown loss function '{name}'. Available: {list(LOSS_FN_MAP.keys())}")
    return LOSS_FN_MAP[name]()


def build_optimizer(name: str, model, lr: float, weight_decay: float):
    """Create optimizer. Supported: adam, adamw, sgd"""
    if name not in OPTIMIZER_MAP:
        raise ValueError(f"Unknown optimizer '{name}'. Available: {list(OPTIMIZER_MAP.keys())}")
    return OPTIMIZER_MAP[name](model.parameters(), lr=float(lr), weight_decay=float(weight_decay))


def build_lr_scheduler(cfg, optimizer):
    """Create learning rate scheduler from config."""
    return LRScheduler(cfg, optimizer)
