'''
Image transform builders.
'''
from torchvision import transforms


def get_transform(cfg):
    """Create transform from config."""
    t = cfg['transform']
    return transforms.Compose([
        transforms.Resize((t['image_size'], t['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=t['normalize_mean'], std=t['normalize_std'])
    ])
