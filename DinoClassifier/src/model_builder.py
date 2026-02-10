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

def build_loss_fn(name: str):
    """Create loss function from name."""
    return LOSS_FN_MAP[name]()


def build_optimizer(name: str, model, lr: float, weight_decay: float = 0.0):
    """Create optimizer from name."""
    return OPTIMIZER_MAP[name](model.parameters(), lr=float(lr), weight_decay=float(weight_decay))


def build_scheduler(optimizer, warmup_epochs: int = 0, decay_type: str = 'none',
                    total_epochs: int = 100, step_size: int = 10, gamma: float = 0.1, eta_min: float = 0.0):
    eta_min = float(eta_min)
    gamma = float(gamma)
    """Create scheduler with optional warmup and decay.

    Args:
        optimizer: The optimizer
        warmup_epochs: Number of warmup epochs (0 = disable)
        decay_type: 'step', 'cosine', or 'none'
        total_epochs: Total training epochs (for cosine annealing T_max)
        step_size: Epochs between LR drops (for StepLR)
        gamma: LR multiplier at each step (for StepLR)
        eta_min: Minimum LR (for CosineAnnealingLR)
    """
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


class CustomClassifierModel(nn.Module):
    def __init__(self, backbone_model, backbone_model_output_dim, freeze_backbone=False):
        super().__init__()
        self.feature_extractor = backbone_model
        self.mlp_head = nn.Sequential(
                        nn.LayerNorm(backbone_model_output_dim),
                        nn.Linear(backbone_model_output_dim, 128),
                        nn.GELU(),
                        nn.Dropout(0.2),
                        nn.Linear(128, 1)
        )
        # Model: logits
        # Loss: BCEWithLogits
        #           └ sigmoid inside
    
        if freeze_backbone:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
    
    def forward(self, x):
        return self.mlp_head(self.feature_extractor(x))


if __name__== "__main__":
    # load the backbone model
    # reference: https://github.com/facebookresearch/dinov3
    device = torch.device("cuda:0")
    REPO_DIR = "/home/jingjie/dev/dino/dinov3"
    CHECKPOINT_PATH = "/home/jingjie/dev/dino/DinoClassifier/models/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
    
    dinov3_vits16 = torch.hub.load(REPO_DIR, 'dinov3_vits16', source='local', weights=CHECKPOINT_PATH)
    
    # instantiate a custom classifier model with the loaded backbone
    model = CustomClassifierModel(
                backbone_model=dinov3_vits16,
                backbone_model_output_dim=384,
                freeze_backbone=False
                ).to(device) # move to device

    model.eval() # set to eval mode

    # prepare dummy input to test forward pass
    # batchsize of 2, 3 rgb channels, image is 224*224
    dummy_input = torch.randn(2, 3, 512, 512).to(device) # also move to device, ensure same location with model

    # run forward pass and visualize the outputs
    with torch.no_grad():
        output = model(dummy_input)
        probs = torch.sigmoid(output)

    print("Output shape:", output.shape)
    print("Output:", output)
    print(probs)