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
    ReduceLROnPlateau
)


class LRScheduler:
    """Unified learning rate scheduler with warmup support."""

    def __init__(self, cfg, optimizer):
        self._warmup_epochs = cfg.scheduler.get('warmup_epochs', 0)
        self._warmup_scheduler = None
        self._decay_scheduler = None
        self._is_plateau = False
        self._current_epoch = 0

        self._build(cfg, optimizer)

    def _build(self, cfg, optimizer):
        """Build the underlying PyTorch scheduler."""
        decay_type = cfg.scheduler.get('decay_type', None)
        total_epochs = cfg.training.epochs

        # Warmup scheduler
        if self._warmup_epochs > 0:
            def lr_lambda(epoch):
                return min(1.0, (epoch + 1) / self._warmup_epochs)
            self._warmup_scheduler = LambdaLR(optimizer, lr_lambda)

        # Decay scheduler
        if decay_type == 'step':
            step_size = cfg.scheduler.step_size
            gamma = cfg.scheduler.get('gamma', 0.1)
            if step_size > 0:
                self._decay_scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

        elif decay_type == 'cosine':
            eta_min = cfg.scheduler.get('eta_min', 1e-7)
            t_max = max(1, total_epochs - self._warmup_epochs)
            self._decay_scheduler = CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)

        elif decay_type == 'plateau':
            patience = cfg.scheduler.get('patience', 10)
            factor = cfg.scheduler.get('factor', 0.1)
            min_lr = cfg.scheduler.get('min_lr', 1e-6)
            threshold = cfg.scheduler.get('min_delta', 1e-4)
            self._decay_scheduler = ReduceLROnPlateau(
                optimizer, mode='min', factor=factor,
                patience=patience, min_lr=min_lr, threshold=threshold
            )
            self._is_plateau = True

    def step(self, metric=None):
        """Step the scheduler. Metric is only used for plateau mode."""
        if self._current_epoch < self._warmup_epochs:
            if self._warmup_scheduler is not None:
                self._warmup_scheduler.step()
        else:
            if self._decay_scheduler is not None:
                if self._is_plateau:
                    self._decay_scheduler.step(metric)
                else:
                    self._decay_scheduler.step()

        self._current_epoch += 1
