'''
Image transform builders.
'''
import torch
from torchvision.transforms import v2

def get_transform(cfg):
    """Create image transform pipeline using torchvision v2."""
    t = cfg['transform']
    to_tensor = v2.ToImage()
    resize = v2.Resize((t['image_size'], t['image_size']), antialias=True)
    to_float = v2.ToDtype(torch.float32, scale=True)
    normalize = v2.Normalize(
        mean=t['normalize_mean'],
        std=t['normalize_std'],
    )
    return v2.Compose([to_tensor, resize, to_float, normalize])

