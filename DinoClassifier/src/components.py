'''
Training components: loss functions, optimizers, and schedulers.
'''
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import LambdaLR, StepLR, CosineAnnealingLR, SequentialLR
from utils.config import get_config

cfg = get_config()

LOSS_FN_MAP = {
    'cross_entropy': nn.CrossEntropyLoss,
    'bce': nn.BCELoss,
    'bce_with_logits': nn.BCEWithLogitsLoss,
    'mse': nn.MSELoss,
}

OPTIMIZER_MAP = {
    'adam': optim.Adam,
    'adamw': optim.AdamW,
    'sgd': optim.SGD,
}

def build_loss_fn():
    """Create loss function from config."""
    return LOSS_FN_MAP[cfg['training']['loss_fn']]()

def build_optimizer(model):
    """Create optimizer from config."""
    name = cfg['training']['optimizer']
    lr = float(cfg['training']['learning_rate'])
    weight_decay = float(cfg['training']['weight_decay'])
    return OPTIMIZER_MAP[name](model.parameters(), lr=lr, weight_decay=weight_decay)

def build_scheduler(optimizer):
    """Create scheduler from config."""
    warmup_epochs = cfg['scheduler']['warmup_epochs']
    decay_type = cfg['scheduler']['decay_type']
    total_epochs = cfg['training']['epochs']
    step_size = cfg['scheduler']['step_size']
    gamma = float(cfg['scheduler']['gamma'])
    eta_min = float(cfg['scheduler']['eta_min'])
    schedulers = []
    milestones = []

    # Warmup scheduler
    if warmup_epochs > 0:
        def lr_lambda(epoch):
            return min(1.0, (epoch + 1) / warmup_epochs)
        schedulers.append(LambdaLR(optimizer, lr_lambda))
        milestones.append(warmup_epochs)

    # Decay scheduler
    if decay_type == 'step' and step_size > 0:
        schedulers.append(StepLR(optimizer, step_size=step_size, gamma=gamma))
    elif decay_type == 'cosine':
        t_max = total_epochs - warmup_epochs  # remaining epochs after warmup
        schedulers.append(CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min))

    # Return appropriate scheduler
    if len(schedulers) == 0:
        return None
    elif len(schedulers) == 1:
        return schedulers[0]
    else:
        return SequentialLR(optimizer, schedulers, milestones)
