import sys
import torch
import torch.nn as nn

sys.path.insert(0, '/mnt3/repo_and_weights/repo/dinov3')

REPO_DIR = '/mnt3/repo_and_weights/repo/dinov3'

WEIGHTS_DIR = '/mnt3/repo_and_weights/weights'

WEIGHTS_MAP = {
    "dinov3_vits16": f"{WEIGHTS_DIR}/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
    "dinov3_vits16plus": f"{WEIGHTS_DIR}/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth",
    "dinov3_vitb16": f"{WEIGHTS_DIR}/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
    "dinov3_vitl16": f"{WEIGHTS_DIR}/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
    "dinov3_vith16plus": f"{WEIGHTS_DIR}/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth",
    "dinov3_vit7b16": f"{WEIGHTS_DIR}/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth",
    "dinov3_convnext_tiny": f"{WEIGHTS_DIR}/dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth",
    "dinov3_convnext_small": f"{WEIGHTS_DIR}/dinov3_convnext_small_pretrain_lvd1689m-296db49d.pth",
    "dinov3_convnext_base": f"{WEIGHTS_DIR}/dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth",
    "dinov3_convnext_large": f"{WEIGHTS_DIR}/dinov3_convnext_large_pretrain_lvd1689m-61fa432d.pth",
}
OUTPUT_DIM = {
    "dinov3_vits16": 384,
    "dinov3_vits16plus":  384,
    "dinov3_vitb16": 768,
    "dinov3_vitl16": 1024,
    "dinov3_vith16plus": 1280,
    "dinov3_vit7b16": 4096,
    "dinov3_convnext_tiny": 768,
    "dinov3_convnext_small": 768,
    "dinov3_convnext_base": 1024,
    "dinov3_convnext_large": 1536,
}

def load_dino_model(cfg) -> tuple[nn.Module, int]:
    """Load a DiNOv3 model with pretrained weights.

    Returns:
        (model, output_dim) tuple
    """
    model_name = cfg['model']['backbone_name']

    if model_name not in WEIGHTS_MAP:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(WEIGHTS_MAP.keys())}")

    weights_path = WEIGHTS_MAP[model_name]

    # Load model architecture without pretrained weights
    model = torch.hub.load(REPO_DIR, model_name, source='local', pretrained=False)

    # Load weights from local file
    state_dict = torch.load(weights_path)
    model.load_state_dict(state_dict)

    return model, OUTPUT_DIM[model_name]
