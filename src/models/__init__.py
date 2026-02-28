'''
Model architectures and builders.
'''
import torch
from .dino import load_dino_model
from .head import classification_head
from .classifier import CustomClassifierModel

def build_classifier_model(cfg, device, backbone_model):
    """Assemble backbone + head into a full model."""
    backbone = backbone_model
    head = classification_head(cfg)
    return CustomClassifierModel(
        backbone=backbone,
        head=head,
        freeze_backbone=cfg['model']['freeze_backbone'],
    ).to(device)

def load_weights_from_checkpoint(model, checkpoint_path, device):
    """Load model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    return model