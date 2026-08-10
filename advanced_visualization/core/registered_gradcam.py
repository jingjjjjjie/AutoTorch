"""Memory-bounded Grad-CAM++ generation driven by a model-local router."""

from __future__ import annotations

import json
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from tqdm import tqdm

from advanced_visualization.core.gradcam_cache import gradcam_cache_candidates
from advanced_visualization.core.heatmap import jet_overlay
from advanced_visualization.core.images import valid_image
from advanced_visualization.core.model_router import ModelRoute, registered_model_routes
from advanced_visualization.models.registry import get_gradcam_engine


WEBP_QUALITY = 80
DEFAULT_CUDA_MEMORY_LIMIT_MB = 2400
TARGETS = ("genuine", "fraud")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _manifest(route: ModelRoute, artifact_dir: Path) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "model_id": route.model_id,
        "router": str(route.router_path),
        "data_dir": str(route.data_dir),
        "artifact_dir": str(artifact_dir),
        "prepared_predictions": str(route.prediction_data),
        "features": str(route.feature_data) if route.feature_data else "",
        "checkpoint": str(route.checkpoint) if route.checkpoint else "",
        "framework": route.framework,
        "engine": route.engine,
        "branch": route.branch,
        "model_name": route.model_name,
        "head_type": route.head_type,
        "image_size": route.image_size,
        "columns": route.columns,
        "gradcam": {
            "method": "gradcam++",
            "targets": list(TARGETS),
            "layers": [
                {
                    "key": layer.key,
                    "label": layer.label,
                    "module": layer.module,
                    "final": layer.final,
                    "max_long_edge": layer.max_long_edge,
                }
                for layer in route.layers
            ],
            "format": "webp",
            "quality": WEBP_QUALITY,
            "preserve_aspect_ratio": True,
            "allow_upscale": False,
            "original_image": "separate_reference",
            "quantitative_saliency_data": False,
        },
        "review": {
            "prepared_gradcam_layers": [
                layer.key for layer in route.final_layers
            ],
        },
    }


def initialize_artifact(route: ModelRoute, artifact_root: Path | None = None) -> Path:
    artifact_dir = (artifact_root / route.model_id) if artifact_root else route.artifact_dir
    (artifact_dir / "gradcam").mkdir(parents=True, exist_ok=True)
    (artifact_dir / "features").mkdir(parents=True, exist_ok=True)
    _atomic_json(artifact_dir / "visualization_manifest.json", _manifest(route, artifact_dir))
    return artifact_dir


def _model_config(route: ModelRoute):
    if route.checkpoint is None:
        raise ValueError(f"{route.model_id} router did not provide a checkpoint.")
    from advanced_visualization.core.config import ModelRunConfig

    return ModelRunConfig(
        key=route.model_id,
        checkpoint=route.checkpoint,
        model_name=route.model_name,
        head_type=route.head_type,
        image_size=route.image_size,
        image_column=route.columns.get("image", ""),
        gradcam_engine=route.engine,
        prediction_column=route.columns.get("prediction", ""),
    )


def _module_at_path(model: torch.nn.Module, path: str) -> torch.nn.Module:
    value: Any = model
    for part in path.split("."):
        value = value[int(part)] if part.isdigit() else getattr(value, part)
    if not isinstance(value, torch.nn.Module):
        raise TypeError(f"Router layer path {path!r} did not resolve to a module.")
    return value


def _configure_cuda_limit() -> None:
    if not torch.cuda.is_available():
        return
    limit_mb = int(
        os.environ.get(
            "AUTOTORCH_CUDA_MEMORY_LIMIT_MB",
            str(DEFAULT_CUDA_MEMORY_LIMIT_MB),
        )
    )
    total = torch.cuda.get_device_properties(0).total_memory
    fraction = min(1.0, (limit_mb * 1024 * 1024) / total)
    torch.cuda.set_per_process_memory_fraction(fraction, device=0)
    torch.cuda.reset_peak_memory_stats(0)


def _base_image(image: Image.Image, max_long_edge: int) -> np.ndarray:
    rendered = image.copy()
    if max(rendered.size) > max_long_edge:
        rendered.thumbnail(
            (max_long_edge, max_long_edge), Image.Resampling.LANCZOS
        )
    return np.asarray(rendered, dtype=np.uint8)


def _save_webp(base: np.ndarray, cam: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    try:
        jet_overlay(base, cam).save(
            temporary,
            format="WEBP",
            quality=WEBP_QUALITY,
            method=4,
        )
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)


def _output_path(
    artifact_dir: Path,
    image_path: Path,
    layer: str,
    target: str,
) -> Path:
    return gradcam_cache_candidates(
        artifact_dir / "gradcam",
        image_path,
        method="gradcam++",
        target=target,
        layer=layer,
    )[0]


def generate_pytorch_layer(
    model_id: str,
    layer_key: str,
    *,
    limit: int | None = None,
    overwrite: bool = False,
    artifact_root: Path | None = None,
    num_shards: int = 1,
    shard_index: int = 0,
) -> dict[str, Any]:
    if num_shards < 1:
        raise ValueError("num_shards must be at least 1.")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("shard_index must be in [0, num_shards).")
    route = registered_model_routes()[model_id]
    if route.framework != "pytorch":
        raise ValueError(f"{model_id} is a {route.framework} route, not PyTorch.")
    layer = route.layer(layer_key)
    image_column = route.columns.get("image", "")
    if not image_column:
        raise ValueError(f"{model_id} router did not provide columns.image.")

    artifact_dir = initialize_artifact(route, artifact_root)
    _configure_cuda_limit()
    engine = get_gradcam_engine(route.engine)
    bundle = engine.load_bundle(_model_config(route))
    model = bundle.model
    device = bundle.device
    model.eval()
    model.requires_grad_(False)
    target_layer = _module_at_path(model, layer.module)

    frame = pd.read_csv(route.prediction_data, usecols=[image_column], low_memory=False)
    image_paths: list[Path] = []
    seen: set[str] = set()
    invalid_sources = 0
    for raw_path in frame[image_column]:
        image_path = valid_image(raw_path)
        if image_path is None:
            invalid_sources += 1
            continue
        resolved = str(image_path.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        image_paths.append(image_path)
    image_paths = image_paths[shard_index::num_shards]
    if limit is not None:
        image_paths = image_paths[:limit]

    generated = skipped = failed = 0
    state_path = artifact_dir / (
        f"gradcam_generation_state.{layer.key}."
        f"shard-{shard_index:03d}-of-{num_shards:03d}.json"
    )
    progress = tqdm(
        image_paths,
        desc=f"{model_id}:{layer.key}:shard-{shard_index:03d}",
        unit="img",
    )

    for row_number, image_path in enumerate(progress, start=1):
        outputs = {
            target: _output_path(artifact_dir, image_path, layer.key, target)
            for target in TARGETS
        }
        missing_targets = [
            target
            for target, path in outputs.items()
            if overwrite or not path.is_file()
        ]
        if not missing_targets:
            skipped += len(TARGETS)
            continue
        try:
            with Image.open(image_path) as opened:
                transposed = ImageOps.exif_transpose(opened)
                image = (transposed or opened).convert("RGB")
                base = _base_image(image, layer.max_long_edge)
                transformed = bundle.transform(image)

            for target in missing_targets:
                activations: dict[str, torch.Tensor] = {}
                gradients: dict[str, torch.Tensor] = {}

                def forward_hook(_module, _inputs, output):
                    activations["value"] = output

                def backward_hook(_module, _grad_input, grad_output):
                    gradients["value"] = grad_output[0]

                forward_handle = target_layer.register_forward_hook(forward_hook)
                backward_handle = target_layer.register_full_backward_hook(backward_hook)
                try:
                    input_tensor = transformed.unsqueeze(0).to(device)
                    input_tensor.requires_grad_(True)
                    model.zero_grad(set_to_none=True)
                    offload = (
                        torch.autograd.graph.save_on_cpu(pin_memory=True)
                        if device.type == "cuda"
                        else nullcontext()
                    )
                    with torch.enable_grad(), offload:
                        score = engine.score(model, input_tensor).mean()
                        (-score if target == "genuine" else score).backward()
                    activation = activations["value"].detach()
                    gradient = gradients["value"].detach()
                    cam = engine.compute_cam(
                        activation, gradient, method="gradcam++"
                    ).float()
                    height, width = base.shape[:2]
                    resized = F.interpolate(
                        cam,
                        size=(height, width),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze()
                    resized = (resized - resized.min()) / (
                        resized.max() - resized.min() + 1e-8
                    )
                    _save_webp(base, resized.cpu().numpy().copy(), outputs[target])
                    generated += 1
                finally:
                    forward_handle.remove()
                    backward_handle.remove()
            if row_number % 100 == 0:
                peak_mb = (
                    round(torch.cuda.max_memory_reserved(0) / 1024 / 1024, 1)
                    if device.type == "cuda"
                    else 0.0
                )
                _atomic_json(
                    state_path,
                    {
                        "model_id": model_id,
                        "layer": layer.key,
                        "num_shards": num_shards,
                        "shard_index": shard_index,
                        "processed_rows": row_number,
                        "generated": generated,
                        "skipped": skipped,
                        "failed": failed,
                        "invalid_sources": invalid_sources,
                        "torch_peak_reserved_mib": peak_mb,
                    },
                )
        except Exception as exc:
            failed += len(missing_targets)
            progress.write(
                f"{model_id}:{layer.key}: {image_path}: "
                f"{type(exc).__name__}: {exc}"
            )
        progress.set_postfix(generated=generated, skipped=skipped, failed=failed)

    peak_mb = (
        round(torch.cuda.max_memory_reserved(0) / 1024 / 1024, 1)
        if device.type == "cuda"
        else 0.0
    )
    result = {
        "model_id": model_id,
        "layer": layer.key,
        "num_shards": num_shards,
        "shard_index": shard_index,
        "source_images": len(image_paths),
        "generated": generated,
        "skipped": skipped,
        "failed": failed,
        "invalid_sources": invalid_sources,
        "torch_peak_reserved_mib": peak_mb,
        "complete": True,
    }
    _atomic_json(state_path, result)
    return result
