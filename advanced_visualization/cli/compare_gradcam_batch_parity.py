"""Compare sequential and correctly batched Grad-CAM++ outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageOps

from advanced_visualization.core.heatmap import jet_overlay
from advanced_visualization.core.images import valid_image
from advanced_visualization.core.model_router import registered_model_routes
from advanced_visualization.core.registered_gradcam import _model_config, _module_at_path
from advanced_visualization.models.registry import get_gradcam_engine


TARGETS = ("fraud", "genuine")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-id",
        default="Ex8point2_UniRepLKNet_T_legacy_v1_512_ori_epoch10",
    )
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _capture(model, engine, target_layer, inputs, target):
    captured = {}

    def forward_hook(_module, _inputs, output):
        captured["activation"] = output

    def backward_hook(_module, _grad_input, grad_output):
        captured["gradient"] = grad_output[0]

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)
    try:
        inputs = inputs.requires_grad_(True)
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            scores = engine.score(model, inputs)
            objective = scores.sum()
            if target == "genuine":
                objective = -objective
            objective.backward()
        return engine.compute_cam(
            captured["activation"].detach(),
            captured["gradient"].detach(),
            method="gradcam++",
        ).float().cpu()
    finally:
        forward_handle.remove()
        backward_handle.remove()


def _normalize(cam: torch.Tensor) -> np.ndarray:
    values = cam.squeeze().numpy().astype(np.float64)
    return (values - values.min()) / (values.max() - values.min() + 1e-8)


def _render(base: np.ndarray, cam: torch.Tensor) -> tuple[Image.Image, np.ndarray]:
    resized = F.interpolate(
        cam.unsqueeze(0) if cam.ndim == 3 else cam,
        size=base.shape[:2],
        mode="bilinear",
        align_corners=False,
    ).squeeze()
    normalized = _normalize(resized)
    return jet_overlay(base, normalized).convert("RGB"), normalized


def _panel(image: Image.Image, max_edge: int = 512) -> Image.Image:
    result = image.copy()
    result.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (max_edge, max_edge), "#10151c")
    canvas.paste(result, ((max_edge - result.width) // 2, (max_edge - result.height) // 2))
    return canvas


def _montage(
    base: np.ndarray,
    rendered: dict[str, dict[str, Image.Image]],
    normalized: dict[str, dict[str, np.ndarray]],
    batch_size: int,
) -> Image.Image:
    size = 512
    header = 44
    columns = ("Original", "Batch 1", f"Batch {batch_size}", "Difference x100")
    rows = TARGETS
    canvas = Image.new("RGB", (size * len(columns), (size + header) * len(rows)), "#10151c")
    draw = ImageDraw.Draw(canvas)
    original = _panel(Image.fromarray(base, "RGB"), size)
    for row_index, target in enumerate(rows):
        y = row_index * (size + header)
        diff = np.abs(normalized[target]["batch1"] - normalized[target]["batch8"])
        difference = Image.fromarray(
            np.clip(diff * 100.0 * 255.0, 0, 255).astype(np.uint8),
            "L",
        ).convert("RGB")
        panels = (
            original,
            _panel(rendered[target]["batch1"], size),
            _panel(rendered[target]["batch8"], size),
            _panel(difference, size),
        )
        for column_index, (label, panel) in enumerate(zip(columns, panels)):
            x = column_index * size
            canvas.paste(panel, (x, y + header))
            draw.text((x + 12, y + 14), f"{target.title()} | {label}", fill="white")
    return canvas


def main() -> None:
    args = parse_args()
    if args.batch_size < 2 or args.count % args.batch_size:
        raise ValueError("count must be divisible by a batch size of at least 2.")
    route = registered_model_routes()[args.model_id]
    layer = route.final_layers[0]
    image_column = route.columns["image"]
    frame = pd.read_csv(route.prediction_data, usecols=[image_column], low_memory=False)
    paths = []
    seen = set()
    for raw_path in frame[image_column]:
        image_path = valid_image(raw_path)
        if image_path is None:
            continue
        resolved = str(image_path.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        paths.append(image_path)
        if len(paths) == args.count:
            break
    if len(paths) != args.count:
        raise RuntimeError(f"Found only {len(paths)} valid unique images.")

    engine = get_gradcam_engine(route.engine)
    bundle = engine.load_bundle(_model_config(route))
    model = bundle.model.eval()
    model.requires_grad_(False)
    target_layer = _module_at_path(model, layer.module)

    images = []
    bases = []
    tensors = []
    for path in paths:
        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            base_image = image.copy()
            if max(base_image.size) > layer.max_long_edge:
                base_image.thumbnail(
                    (layer.max_long_edge, layer.max_long_edge),
                    Image.Resampling.LANCZOS,
                )
            images.append(image.copy())
            bases.append(np.asarray(base_image, dtype=np.uint8))
            tensors.append(bundle.transform(image))

    batch = torch.stack(tensors).to(bundle.device)
    sequential = {target: [] for target in TARGETS}
    batched = {}
    for target in TARGETS:
        for tensor in tensors:
            sequential[target].append(
                _capture(
                    model,
                    engine,
                    target_layer,
                    tensor.unsqueeze(0).to(bundle.device),
                    target,
                )[0]
            )
        batched[target] = torch.cat(
            [
                _capture(
                    model,
                    engine,
                    target_layer,
                    batch[start : start + args.batch_size].clone(),
                    target,
                )
                for start in range(0, args.count, args.batch_size)
            ],
            dim=0,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "model_id": args.model_id,
        "layer": layer.key,
        "batch_size": args.batch_size,
        "images": [],
    }
    first_rendered = {target: {} for target in TARGETS}
    first_normalized = {target: {} for target in TARGETS}
    for image_index, path in enumerate(paths):
        image_metrics = {"image": str(path), "targets": {}}
        for target in TARGETS:
            batch1_native = _normalize(sequential[target][image_index])
            batch8_native = _normalize(batched[target][image_index])
            native_difference = np.abs(batch1_native - batch8_native)
            batch1_overlay, batch1_normalized = _render(
                bases[image_index], sequential[target][image_index]
            )
            batch8_overlay, batch8_normalized = _render(
                bases[image_index], batched[target][image_index]
            )
            image_metrics["targets"][target] = {
                "native_max_abs": float(native_difference.max()),
                "native_mean_abs": float(native_difference.mean()),
                "native_correlation": float(
                    np.corrcoef(batch1_native.ravel(), batch8_native.ravel())[0, 1]
                ),
                "peak_batch1": [
                    int(value)
                    for value in np.unravel_index(batch1_native.argmax(), batch1_native.shape)
                ],
                "peak_batch8": [
                    int(value)
                    for value in np.unravel_index(batch8_native.argmax(), batch8_native.shape)
                ],
            }
            if image_index == 0:
                batch1_overlay.save(
                    args.output_dir / f"batch1_{target}.webp",
                    format="WEBP",
                    quality=80,
                    method=4,
                )
                batch8_overlay.save(
                    args.output_dir / f"batch{args.batch_size}_{target}.webp",
                    format="WEBP",
                    quality=80,
                    method=4,
                )
                first_rendered[target]["batch1"] = batch1_overlay
                first_rendered[target]["batch8"] = batch8_overlay
                first_normalized[target]["batch1"] = batch1_normalized
                first_normalized[target]["batch8"] = batch8_normalized
        metrics["images"].append(image_metrics)

    maxima = [
        target["native_max_abs"]
        for image in metrics["images"]
        for target in image["targets"].values()
    ]
    means = [
        target["native_mean_abs"]
        for image in metrics["images"]
        for target in image["targets"].values()
    ]
    metrics["summary"] = {
        "comparisons": len(maxima),
        "maximum_native_absolute_difference": max(maxima),
        "mean_native_absolute_difference": float(np.mean(means)),
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    _montage(bases[0], first_rendered, first_normalized, args.batch_size).save(
        args.output_dir / "comparison.webp",
        format="WEBP",
        quality=90,
        method=4,
    )
    print(json.dumps(metrics["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
