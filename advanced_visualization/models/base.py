"""Contracts for model-specific visualization preparation."""
from __future__ import annotations

from pathlib import Path
from typing import Protocol

import torch

from advanced_visualization.core.config import ModelRunConfig


class GradcamEngine(Protocol):
    """Interface implemented by model-specific Grad-CAM builders."""

    name: str

    def load_bundle(self, config: ModelRunConfig) -> dict:
        """Load model, transform, target layer, and device."""

    def score(self, model: torch.nn.Module, input_tensor: torch.Tensor) -> torch.Tensor:
        """Return the model score used for Grad-CAM backpropagation."""

    def compute_cam(self, activation: torch.Tensor, gradient: torch.Tensor) -> torch.Tensor:
        """Convert activation and gradient tensors into raw CAM tensors."""

    def generate(self, config: ModelRunConfig, image_path: Path) -> Path:
        """Generate or return a prepared Grad-CAM overlay for one image."""
