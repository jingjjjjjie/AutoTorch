'''
Model architectures and builders.
'''
import torch
from .head import classification_head
from .classifier import CustomClassifierModel

def load_backbone(cfg, load_weights=True):
    """Load a backbone from local repository."""
    if load_weights:
        return torch.hub.load(
            cfg['model']['repo_dir'],
            cfg['model']['backbone_name'],
            source='local',
            weights=cfg['model']['checkpoint_path'],
        )
    else:
        # When not loading pretrained weights (e.g., loading from checkpoint later),
        # set pretrained=False to skip weight loading entirely
        return torch.hub.load(
            cfg['model']['repo_dir'],
            cfg['model']['backbone_name'],
            source='local',
            pretrained=False,
        )

def build_classifier_model(cfg, device, load_weights=True):
    """Assemble backbone + head into a full model."""
    backbone = load_backbone(cfg, load_weights)
    head = classification_head(cfg)
    return CustomClassifierModel(
        backbone=backbone,
        head=head,
        freeze_backbone=cfg['model']['freeze_backbone'],
    ).to(device)

def load_weights_from_checkpoint(model, checkpoint_path, device):
    """Load model from checkpoint."""
    # model = build_classifier_model(cfg, device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    #model.eval()
    return model