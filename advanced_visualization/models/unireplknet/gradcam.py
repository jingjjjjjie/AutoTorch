"""Grad-CAM implementation for UniRepLKNet AutoTorch classifier models."""
from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from advanced_visualization.core.config import DEFAULT_GRADCAM_ROOT, DEFAULT_MEAN, DEFAULT_STD, SRC_ROOT, ModelRunConfig
from advanced_visualization.core.gradcam_cache import gradcam_cache_candidates
from advanced_visualization.core.heatmap import jet_overlay

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data.idfraud.transforms import build_transform
from models import build_model


class UniRepLKNetGradcamEngine:
    """Grad-CAM engine for UniRepLKNet backbone + AutoTorch classifier-head models."""

    name = "unireplknet"

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._bundles: dict[str, dict] = {}

    def load_bundle(self, config: ModelRunConfig) -> dict:
        with self._lock:
            if config.key in self._bundles:
                return self._bundles[config.key]

            if not config.checkpoint.exists():
                raise FileNotFoundError(f"Grad-CAM checkpoint missing: {config.checkpoint}")

            device = self._device()
            model = build_model(
                model_name=config.model_name,
                device=device,
                task="classification",
                head_type=config.head_type,
                freeze_backbone=False,
            )
            self._load_state_dict(model, config.checkpoint, device)
            model.eval()
            transform = build_transform(
                image_size=config.image_size,
                normalize_mean=DEFAULT_MEAN,
                normalize_std=DEFAULT_STD,
                version=config.transform_version,
            )
            bundle = {
                "model": model,
                "transform": transform,
                "target_layer": model.feature_extractor.stages[-1],
                "device": device,
            }
            self._bundles[config.key] = bundle
            return bundle

    def score(self, model: torch.nn.Module, input_tensor: torch.Tensor) -> torch.Tensor:
        features = model.feature_extractor(input_tensor)
        head_sequence = getattr(model.mlp_head, "fc", None)
        if head_sequence is None:
            head_sequence = getattr(model.mlp_head, "head", None)

        if isinstance(head_sequence, torch.nn.Sequential) and len(head_sequence) > 0:
            layers = list(head_sequence.children())
            if isinstance(layers[-1], torch.nn.Sigmoid):
                score = features
                for layer in layers[:-1]:
                    score = layer(score)
                return score.squeeze()

        return model.mlp_head(features).squeeze()

    def compute_cam(self, activation: torch.Tensor, gradient: torch.Tensor, method: str = "gradcam") -> torch.Tensor:
        if method in {"gradcam++", "gradcampp"}:
            positive_gradients = torch.relu(gradient)
            gradients_power_2 = gradient.pow(2)
            gradients_power_3 = gradients_power_2 * gradient
            denominator = 2.0 * gradients_power_2 + (activation * gradients_power_3).sum(dim=(2, 3), keepdim=True)
            alpha = gradients_power_2 / (denominator + 1e-8)
            weights = (alpha * positive_gradients).sum(dim=(2, 3), keepdim=True)
            return torch.relu((weights * activation).sum(dim=1, keepdim=True))
        weights = gradient.mean(dim=(2, 3), keepdim=True)
        return torch.relu((weights * activation).sum(dim=1, keepdim=True))

    def generate(self, config: ModelRunConfig, image_path: Path, target: str = "fraud") -> Path:
        if not image_path.is_file():
            raise FileNotFoundError(f"Image missing: {image_path}")

        output_root = self._artifact_root(config)
        output_root.mkdir(parents=True, exist_ok=True)
        if target not in {"fraud", "genuine"}:
            raise ValueError(f"Unsupported CAM target: {target}")
        output_path = gradcam_cache_candidates(output_root, image_path, target=target)[0]
        if output_path.exists():
            return output_path

        bundle = self.load_bundle(config)
        model = bundle["model"]
        transform = bundle["transform"]
        target_layer = bundle["target_layer"]
        device = bundle["device"]
        activations = {}
        gradients = {}

        def forward_hook(_module, _inputs, output):
            activations["value"] = output

        def backward_hook(_module, _grad_input, grad_output):
            gradients["value"] = grad_output[0]

        forward_handle = target_layer.register_forward_hook(forward_hook)
        backward_handle = target_layer.register_full_backward_hook(backward_hook)
        try:
            image = Image.open(image_path).convert("RGB")
            input_tensor = transform(image).unsqueeze(0).to(device)
            model.zero_grad(set_to_none=True)
            score = self.score(model, input_tensor).mean()
            if target == "genuine":
                score = -score
            score.backward()

            activation = activations["value"].detach()
            gradient = gradients["value"].detach()
            cam = self.compute_cam(activation, gradient)
            cam = F.interpolate(cam, size=(image.height, image.width), mode="bilinear", align_corners=False)
            cam = cam.squeeze().float()
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            tmp_output = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
            jet_overlay(np.asarray(image.convert("RGB")), cam.detach().cpu().numpy()).save(
                tmp_output, format="PNG", compress_level=3
            )
            os.replace(tmp_output, output_path)
            return output_path
        finally:
            if "tmp_output" in locals():
                tmp_output.unlink(missing_ok=True)
            forward_handle.remove()
            backward_handle.remove()

    def _artifact_root(self, config: ModelRunConfig) -> Path:
        checkpoint = config.checkpoint
        if checkpoint.parent.name == "checkpoints":
            return checkpoint.parent.parent / "gradcam"
        return DEFAULT_GRADCAM_ROOT / config.key

    def _device(self) -> torch.device:
        requested = os.environ.get("AUTOTORCH_GRADCAM_DEVICE")
        if requested:
            return torch.device(requested)
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def _load_state_dict(self, model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> None:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        normalized = {}
        for key, value in state_dict.items():
            if key.startswith("module."):
                key = key[len("module.") :]
            normalized[key] = value
        model.load_state_dict(normalized, strict=True)
