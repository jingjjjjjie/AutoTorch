'''
Image transform builder from dinov3's repository
'''
import torch
from typing import Tuple
from torchvision.transforms import v2


def get_transform(image_size: int,
                  normalize_mean: Tuple[float, ...],
                  normalize_std: Tuple[float, ...]) -> v2.Compose:
    """
    Args:
        image_size: Target size for resizing (square).
        normalize_mean: Mean values for normalization (per channel).
        normalize_std: Std values for normalization (per channel).
    """
    return v2.Compose([
        v2.ToImage(),
        v2.Resize((image_size, image_size), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=list(normalize_mean), std=list(normalize_std)),
    ])

