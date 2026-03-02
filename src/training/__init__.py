'''
Training components: loss, optimizers, callbacks, and trainers.
'''
from .loss import LOSS_FN_MAP
from .optimizers import OPTIMIZER_MAP
from .lr_scheduler import LRScheduler

def build_loss_fn(cfg):
    """Create loss function from config.

    Supported: cross_entropy, bce, bce_with_logits, mse
    """
    name = cfg['training']['loss_fn']
    if name not in LOSS_FN_MAP:
        raise ValueError(f"Unknown loss function '{name}'. Available: {list(LOSS_FN_MAP.keys())}")
    return LOSS_FN_MAP[name]()


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


def build_lr_scheduler(cfg, optimizer):
    """Create learning rate scheduler from config."""
    return LRScheduler(cfg, optimizer)
