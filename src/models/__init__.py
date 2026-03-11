'''
Model architectures and builders.
'''
import torch
from .head import ClassificationHeadV1
from .dino import load_dino_model
from .classifier import CustomClassifierModel

HEAD_MAP = {
    'v1': ClassificationHeadV1,
}

def build_head(head_type: str, input_dim: int):
    """Create classification head."""
    if head_type not in HEAD_MAP:
        raise ValueError(f"Unknown head_type '{head_type}'. Available: {list(HEAD_MAP.keys())}")
    return HEAD_MAP[head_type](input_dim)

def build_classifier_model(device, backbone_model, input_dim: int, head_type: str = 'v1', freeze_backbone: bool = True):
    """Assemble backbone + head into a full model."""
    head = build_head(head_type, input_dim)
    return CustomClassifierModel(
        backbone=backbone_model,
        head=head,
        freeze_backbone=freeze_backbone,
    ).to(device)

def load_weights_from_checkpoint(model, checkpoint_path, device):
    """Load model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    return model