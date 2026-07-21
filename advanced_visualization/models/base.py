"""Contracts for model-specific visualization preparation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

import torch
from PIL import Image

from advanced_visualization.core.config import ModelRunConfig


ImageTransform = Callable[[Image.Image], torch.Tensor]


@dataclass(frozen=True)
class GradcamBundle:
    """Loaded model resources shared by feature extraction and CAM generation."""

    model: torch.nn.Module
    transform: ImageTransform
    target_layer: torch.nn.Module
    device: torch.device


class GradcamEngine(Protocol):
    """Interface implemented by model-specific Grad-CAM builders."""

    name: str

    def load_bundle(self, config: ModelRunConfig) -> GradcamBundle:
        """Load model, transform, target layer, and device."""

    def score(self, model: torch.nn.Module, input_tensor: torch.Tensor) -> torch.Tensor:
        """Return the model score used for Grad-CAM backpropagation."""

    def compute_cam(
        self,
        activation: torch.Tensor,
        gradient: torch.Tensor,
        method: str = "gradcam",
    ) -> torch.Tensor:
        """Convert activation and gradient tensors into raw CAM tensors."""

    def generate(
        self, config: ModelRunConfig, image_path: Path, target: str = "fraud"
    ) -> Path:
        """Generate or return a prepared Grad-CAM overlay for one image."""
