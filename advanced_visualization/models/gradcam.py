"""Compatibility API for Grad-CAM generation.

New model-specific implementations belong in `advanced_visualization.models`.
"""
from __future__ import annotations

from pathlib import Path

import torch

from advanced_visualization.core.config import ModelRunConfig, all_model_runs
from advanced_visualization.models.registry import get_gradcam_engine


def engine_for_config(config: ModelRunConfig):
    return get_gradcam_engine(config.gradcam_engine)


def config_for_key(config_key: str) -> ModelRunConfig:
    model_runs = all_model_runs()
    if config_key not in model_runs:
        raise ValueError(f"No Grad-CAM model config for {config_key}")
    return model_runs[config_key]


def load_gradcam_bundle(config_key: str) -> dict:
    config = config_for_key(config_key)
    return engine_for_config(config).load_bundle(config)


def gradcam_score(model: torch.nn.Module, input_tensor: torch.Tensor, config_key: str | None = None) -> torch.Tensor:
    if config_key is None:
        config_key = next(iter(all_model_runs()))
    config = config_for_key(config_key)
    return engine_for_config(config).score(model, input_tensor)


def compute_cam(activation: torch.Tensor, gradient: torch.Tensor, config_key: str | None = None, method: str = "gradcam") -> torch.Tensor:
    if config_key is None:
        config_key = next(iter(all_model_runs()))
    config = config_for_key(config_key)
    return engine_for_config(config).compute_cam(activation, gradient, method=method)


def generate_gradcam(config_key: str, image_path: Path) -> Path:
    config = config_for_key(config_key)
    return engine_for_config(config).generate(config, image_path)
