import sys
import timm
import torch
import torch.nn as nn

REPO_DIR = '/mnt3/repo_and_weights/repo/UniRepLKNet'
WEIGHTS_DIR = '/mnt3/repo_and_weights/weights/unireplknet'

sys.path.insert(0, REPO_DIR)
from unireplknet import *

# more variants to be implemented, refer to https://github.com/AILab-CVC/UniRepLKNet?tab=readme-ov-file
# the link to some weights are broken, refer to https://huggingface.co/DingXiaoH/UniRepLKNet/tree/main for downloads

WEIGHTS_MAP = {
    "unireplknet_n": f"{WEIGHTS_DIR}/unireplknet_n_in1k_224_acc81.64.pth",
    "unireplknet_p": f"{WEIGHTS_DIR}/unireplknet_p_in1k_224_acc80.23.pth",
    "unireplknet_t": f"{WEIGHTS_DIR}/unireplknet_t_in1k_224_acc83.21.pth",
    "unireplknet_s": f"{WEIGHTS_DIR}/unireplknet_s_in1k_224_acc83.91.pth",
    # ImageNet-22K Pretrained Weights
    'unireplknet_s_in22k': f"{WEIGHTS_DIR}/unireplknet_s_in22k_pretrain.pth",
    'unireplknet_b_in22k': f"{WEIGHTS_DIR}/unireplknet_b_in22k_pretrain.pth"

}

OUTPUT_DIM = { 
    # ImageNet-1K Pretrained Weights
    "unireplknet_p":  512,
    "unireplknet_n":  640,
    "unireplknet_t":  640,
    "unireplknet_s":  768,
    # ImageNet-22K Pretrained Weights
    "unireplknet_s_in22k": 768,
    "unireplknet_b_in22k": 1024
}

def load_unireplknet_model(model_name: str) -> tuple[nn.Module, int]:
    """Load a UniRepLKNet model with pretrained weights.

    Returns:
        (model, output_dim) tuple
    """
    if model_name not in WEIGHTS_MAP:
        raise ValueError(f"Unknown UniRepLKNet model: {model_name}. Available: {list(WEIGHTS_MAP.keys())}")

    arch_name = model_name.replace("_in22k", "")
    model = timm.create_model(arch_name, num_classes=1000)

    weights_path = WEIGHTS_MAP[model_name]
    checkpoint = torch.load(weights_path, weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    # Remove head weights to avoid size mismatch (we replace head with Identity anyway)
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith("head.")}
    model.load_state_dict(state_dict, strict=False)

    model.head = nn.Identity()

    return model, OUTPUT_DIM[model_name]
