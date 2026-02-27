'''
Learning rate scheduler with optional warmup.

Supports:
- Linear warmup: gradually increases LR from 0 to target

- Learning rate decay(step or cosine)
    Step decay: reduces LR by gamma every step_size epochs
    Cosine decay: smoothly decreases LR following cosine curve
'''
from torch.optim.lr_scheduler import LambdaLR, StepLR, CosineAnnealingLR, SequentialLR


def lr_scheduler(cfg, optimizer):
    """Create learning rate scheduler from config.

    Supports warmup + decay (step or cosine).
    Returns None if no scheduler is configured.
    """
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
