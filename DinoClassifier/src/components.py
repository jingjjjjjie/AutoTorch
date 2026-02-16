'''
Training components: loss functions, optimizers, and schedulers.
'''
import os
import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import LambdaLR, StepLR, CosineAnnealingLR, SequentialLR
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

def build_loss_fn(cfg):
    """Create loss function from config."""
    return LOSS_FN_MAP[cfg['training']['loss_fn']]()

def build_optimizer(cfg, model):
    """Create optimizer from config."""
    name = cfg['training']['optimizer']
    lr = float(cfg['training']['learning_rate'])
    weight_decay = float(cfg['training']['weight_decay'])
    return OPTIMIZER_MAP[name](model.parameters(), lr=lr, weight_decay=weight_decay)

def build_scheduler(cfg, optimizer):
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


class EarlyStopping:
    def __init__(self, patience=5, delta=0, verbose=False):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.best_loss = None
        self.no_improvement_count = 0
        self.stop_training = False

    def check_early_stop(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1
            if self.no_improvement_count >= self.patience:
                self.stop_training = True
                if self.verbose:
                    print("Stopping early as no improvement has been observed.")


class Checkpoint:
    """Saves model and optimizer state at the end of each epoch."""

    def __init__(self, save_dir, save_name):
        self.ckpt_dir = os.path.join(save_dir, save_name.replace('.pth', ''), 'checkpoints')
        os.makedirs(self.ckpt_dir, exist_ok=True)

    def save(self, model, optimizer, epoch, val_loss):
        state = {
            'epoch': epoch,
            'model_state_dict': model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
        }
        path = os.path.join(self.ckpt_dir, f'epoch_{epoch+1}.pt')
        torch.save(state, path)


def build_checkpoint(cfg):
    """Create Checkpoint from config, or None if disabled."""
    ckpt = cfg.get('checkpoint', {})
    if not ckpt.get('enabled', False):
        return None
    return Checkpoint(
        save_dir=cfg['model']['save_dir'],
        save_name=cfg['model']['save_name'],
    )


def build_early_stopping(cfg):
    """Create EarlyStopping from config, or None if disabled."""
    es = cfg.get('early_stopping', {})
    if not es.get('enabled', False):
        return None
    return EarlyStopping(patience=es['patience'], delta=es['delta'], verbose=es['verbose'])
