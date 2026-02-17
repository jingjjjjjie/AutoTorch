'''
Model architectures and builders.
'''
import torch
from .head import classification_head
from .classifier import CustomClassifierModel

def load_backbone(cfg):
    """Load a backbone from local repository."""
    return torch.hub.load(
        cfg['model']['repo_dir'],
        cfg['model']['backbone_name'],
        source='local',
        weights=cfg['model']['checkpoint_path'],
    )

def build_classifier_model(cfg, device):
    """Assemble backbone + head into a full model."""
    backbone = load_backbone(cfg)
    head = classification_head(cfg)
    return CustomClassifierModel(
        backbone=backbone,
        head=head,
        freeze_backbone=cfg['model']['freeze_backbone'],
    ).to(device)
