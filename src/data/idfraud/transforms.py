import torch
from typing import Tuple
from torchvision.transforms import v2
import torch.nn.functional as F
import torchvision.transforms.functional as TF


def build_transform(image_size: int,
                    normalize_mean: Tuple[float, ...],
                    normalize_std: Tuple[float, ...],
                    version: str = 'v1') -> v2.Compose:
    """Dispatcher that returns the right transform pipeline from config.
    Args:
        version: 'v1' for simple distortion resize, 'v2' for center-crop + resize + pad.
    """
    if version == 'v1':
        return _get_transform(image_size, normalize_mean, normalize_std)
    elif version == 'v2':
        return _get_transform_v2(image_size, normalize_mean, normalize_std)
    else:
        raise ValueError(f"Unknown transform version '{version}'. Supported: 'v1', 'v2'.")



class Center_Crop_Resize_and_Pad:
    def __init__(self, target_height, target_width):
        self.target_height = target_height
        self.target_width = target_width

    def __call__(self, image):
        # image: torch.Tensor [C, H, W] — received after v2.ToImage()
        img_h, img_w = image.shape[-2], image.shape[-1]

        # img_h < img_w, for cropped images (landscape)
        # img_h = img_w, for square images
        # img_h > img_w, for original images (portrait)
        if img_h > img_w:
            image = _center_crop_image_torch(image, img_h, img_w)

        return _pad_to_square(image, self.target_height, self.target_width)


def _get_transform(image_size: int,
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

def _get_transform_v2(image_size: int,
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
        Center_Crop_Resize_and_Pad(image_size, image_size),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=list(normalize_mean), std=list(normalize_std)),
    ])


# utility functions for _get_transform_v2
def _center_crop_image_torch(image, img_h, img_w):
    size = min(img_h, img_w) # Crop a centered square using the shorter side
    left = (img_w - size) // 2
    top = (img_h - size) // 2
    image = TF.crop(image, top, left, size, size) # crop a center square region based on the image width
    return image

def _pad_to_square(image, target_height, target_width):
    img_h, img_w = image.shape[-2], image.shape[-1] # we need to recalculate this because the dimensions may change after center crop 

    scale = min(target_height / img_h, target_width / img_w)
    new_h, new_w = int(img_h * scale), int(img_w * scale)
    image = TF.resize(image, [new_h, new_w], antialias=True)

    pad_h = target_height - new_h
    pad_w = target_width - new_w
    pad_top, pad_bottom = pad_h // 2, pad_h - pad_h // 2
    pad_left, pad_right = pad_w // 2, pad_w - pad_w // 2

    image = F.pad(image, [pad_left, pad_right, pad_top, pad_bottom], mode='constant', value=0)

    return image 