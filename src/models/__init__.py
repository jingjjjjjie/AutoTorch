'''
Model architectures and builders.
'''
import torch
from .backbones import load_backbone, BACKBONE_LOADERS
from .heads import build_head, HEAD_MAP
from .architectures import CustomClassifierModel

# Task -> Architecture mapping
TASK_TO_ARCHITECTURE_MAP = {
    'classification': CustomClassifierModel,
}


def build_model(model_name: str, device, 
                task: str = 'classification',
                head_type: str = 'v1', 
                freeze_backbone: bool = False):
    """
    Builds a complete model(architecture) from backbone + head for a given task.
    Args:
        model_name: Name of backbone 
        device: Device to place model on
        task: Currently only supports 'classification', more tasks comming soon
        head_type: Head variant for the task
        freeze_backbone: Whether to freeze backbone weights
    Returns:
        Model(Architecture) ready for training/inference
    """
    # Checks if tasks is in avaliable tasks
    if task not in TASK_TO_ARCHITECTURE_MAP:
        raise ValueError(f"Unknown task: '{task}'. Available: {list(TASK_TO_ARCHITECTURE_MAP.keys())}")
    # load backbone from model name
    backbone, output_dim = load_backbone(model_name=model_name)
    # build head from task and head type
    head = build_head(task, head_type, output_dim)

    # returns a model/architecture ready for training or inference
    return TASK_TO_ARCHITECTURE_MAP[task](
        backbone=backbone,
        head=head,
        freeze_backbone=freeze_backbone,
    ).to(device)


# load weights from checkpoint (gets the 'model_state_dict') from a given pytorch file (pt, pth, tar)
def load_weights_from_checkpoint(model, checkpoint_path, device):
    """Load model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    return model