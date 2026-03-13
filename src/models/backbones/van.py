import importlib.util
import torch
import torch.nn as nn
import sys

REPO_DIR = '/mnt3/repo_and_weights/repo/VAN-Classification'
WEIGHTS_DIR = '/mnt3/repo_and_weights/weights/van'

# Load VAN module via importlib to avoid namespace collision with our models/
spec = importlib.util.spec_from_file_location("van_repo", f"{REPO_DIR}/models/van.py")
van_module = importlib.util.module_from_spec(spec)

sys.modules["van_repo"] = van_module
spec.loader.exec_module(van_module)

MODEL_BUILDERS = {
    "van_b0": van_module.van_b0,
    "van_b1": van_module.van_b1,
    "van_b2": van_module.van_b2,
    "van_b3": van_module.van_b3,
}

WEIGHTS_MAP = {
    "van_b0": f"{WEIGHTS_DIR}/van_tiny_754.pth.tar",
    "van_b1": f"{WEIGHTS_DIR}/van_small_811.pth.tar",
    "van_b2": f"{WEIGHTS_DIR}/van_base_828.pth.tar",
    "van_b3": f"{WEIGHTS_DIR}/van_large_839.pth.tar",
}

OUTPUT_DIM = {
    "van_b0": 256,
    "van_b1": 512,
    "van_b2": 512,
    "van_b3": 512,
}


def load_van_model(model_name: str) -> tuple[nn.Module, int]:
    """Load a VAN model with pretrained weights.

    Returns:
        (model, output_dim) tuple
    """
    if model_name not in MODEL_BUILDERS:
        raise ValueError(f"Unknown VAN model: {model_name}. Available: {list(MODEL_BUILDERS.keys())}")

    model = MODEL_BUILDERS[model_name](pretrained=False)
    weights_path = WEIGHTS_MAP[model_name]

    # load in the
    checkpoint = torch.load(weights_path, weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict)

    # two options 
    model.head = nn.Identity()
    #model.forward = model.forward_feature

    return model, OUTPUT_DIM[model_name]
