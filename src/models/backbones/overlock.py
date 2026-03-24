import sys
import torch
import torch.nn as nn

REPO_DIR    = '/mnt3/repo_and_weights/repo/OverLoCK/models/'
WEIGHTS_DIR = '/mnt3/repo_and_weights/weights/overlock'

sys.path.insert(0, REPO_DIR)
from overlock import overlock_xt, overlock_t, overlock_s

WEIGHTS_MAP = {
    'overlock_xt': f'{WEIGHTS_DIR}/overlock_xt_in1k_224.pth',
    'overlock_t':  f'{WEIGHTS_DIR}/overlock_t_in1k_224.pth',
    'overlock_s':  f'{WEIGHTS_DIR}/overlock_s_in1k_224.pth',
}

OUTPUT_DIM = {
    'overlock_xt': 1024,  # projection dim (default for all variants)
    'overlock_t':  1024,
    'overlock_s':  1024,
}

_FACTORIES = {
    'overlock_xt': overlock_xt,
    'overlock_t':  overlock_t,
    'overlock_s':  overlock_s,
}


def load_overlock_model(model_name: str) -> tuple[nn.Module, int]:
    """Load an OverLoCK model with pretrained ImageNet-1K weights.

    Returns:
        (model, output_dim) tuple
    """
    if model_name not in WEIGHTS_MAP:
        raise ValueError(f"Unknown OverLoCK model: '{model_name}'. Available: {list(WEIGHTS_MAP.keys())}")

    model = _FACTORIES[model_name](pretrained=False, num_classes=1000)

    checkpoint = torch.load(WEIGHTS_MAP[model_name], weights_only=False)
    state_dict = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))
    model.load_state_dict(state_dict)

    # Remove the classifier (head[-1]) — keep projection features (1024-d)
    model.head[-1] = nn.Identity()

    return model, OUTPUT_DIM[model_name]
