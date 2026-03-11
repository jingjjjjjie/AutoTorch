'''
Image transform builders.
'''
import torch
from torchvision.transforms import v2

def get_transform(transform_cfg):
    """Create image transform pipeline using torchvision v2."""
    to_tensor = v2.ToImage()
    resize = v2.Resize((transform_cfg.image_size, transform_cfg.image_size), antialias=True)
    to_float = v2.ToDtype(torch.float32, scale=True)
    normalize = v2.Normalize(
        mean=list(transform_cfg.normalize_mean),
        std=list(transform_cfg.normalize_std),
    )
    return v2.Compose([to_tensor, resize, to_float, normalize])

