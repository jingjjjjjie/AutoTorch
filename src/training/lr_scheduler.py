'''
Learning rate scheduler with optional warmup.

Supports:
- Linear warmup: gradually increases LR from 0 to target

- Learning rate decay (step, cosine, or plateau)
    Step decay: reduces LR by gamma every step_size epochs
    Cosine decay: smoothly decreases LR following cosine curve
    Plateau: reduces LR when metric stops improving
'''
from torch.optim.lr_scheduler import (
    LambdaLR,
    StepLR,
    CosineAnnealingLR,
    SequentialLR,
    ReduceLROnPlateau
)


class LRScheduler:
    """Unified learning rate scheduler with warmup support."""

    def __init__(self, cfg, optimizer):
        self._scheduler = self._build(cfg, optimizer)
        self._is_plateau = isinstance(self._scheduler, ReduceLROnPlateau)

    def _build(self, cfg, optimizer):
        """Build the underlying PyTorch scheduler."""
        warmup_epochs = cfg['scheduler']['warmup_epochs']
        decay_type = cfg['scheduler']['decay_type']
        total_epochs = cfg['training']['epochs']
        step_size = cfg['scheduler']['step_size']
        gamma = float(cfg['scheduler']['gamma'])
        eta_min = float(cfg['scheduler']['eta_min'])

        # Plateau-specific params
        patience = cfg['scheduler'].get('patience', 10)
        factor = cfg['scheduler'].get('factor', 0.1)
        min_lr = cfg['scheduler'].get('min_lr', 1e-6)

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
            # implements a cosine annealing schedule that gradually reduces the learning rate from an 
            # initial value to a minimum value following a cosine curve
            t_max = max(1, total_epochs - warmup_epochs) # guard, TODO: REVISIT
            schedulers.append(CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min))

        elif decay_type == 'plateau':
            # ReduceLROnPlateau: reduces LR when metric stops improving
            # Plateau?
            # ├─ No  → keep LR
            # └─ Yes → reduce LR
            # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html
            return ReduceLROnPlateau(
                optimizer, # the scheduler directly modifies the optimizer's learning rate
                mode='min',
                factor=factor,
                patience=patience,
                min_lr=min_lr,
                verbose=True
            )

        # Return appropriate scheduler
        if len(schedulers) == 0:
            return None
        elif len(schedulers) == 1:
            return schedulers[0]
        else:
            return SequentialLR(optimizer, schedulers, milestones)

    def step(self, metric=None):
        """Step the scheduler. Metric is only used for plateau mode."""
        if self._scheduler is None:
            return
        if self._is_plateau:
            self._scheduler.step(metric)
        else:
            self._scheduler.step()
