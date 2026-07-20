"""Shared heatmap composition helpers for prepared CAM artifacts."""
from __future__ import annotations

import numpy as np
from PIL import Image


def jet_overlay(base: np.ndarray, cam: np.ndarray) -> Image.Image:
    """Blend a normalized CAM over an RGB image with a JET-style palette."""
    cam = np.nan_to_num(cam, nan=0.0, posinf=1.0, neginf=0.0)
    cam = np.clip(cam, 0.0, 1.0)
    base_f = np.asarray(base, dtype=np.float32)

    # Standard JET ordering: dark blue at the low end, followed by cyan,
    # yellow, and dark red at the highest activations.
    stops = np.array([0.0, 0.125, 0.375, 0.625, 0.875, 1.0], dtype=np.float32)
    red = np.interp(cam, stops, [0, 0, 0, 255, 255, 128])
    green = np.interp(cam, stops, [0, 0, 255, 255, 0, 0])
    blue = np.interp(cam, stops, [128, 255, 255, 0, 0, 0])
    heat = np.stack((red, green, blue), axis=-1).astype(np.float32)

    alpha = np.clip(0.18 + 0.55 * cam[..., None], 0.18, 0.65)
    overlay = base_f * (1.0 - alpha) + heat * alpha
    return Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))
