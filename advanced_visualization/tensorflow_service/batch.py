"""Generate one router-registered VAN Small Grad-CAM++ layer."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

from advanced_visualization.core.model_router import registered_model_routes


WEBP_QUALITY = 80
TARGETS = ("genuine", "fraud")


def _configure_memory() -> None:
    limit_mb = int(os.environ.get("VANSMALL_GPU_MEMORY_LIMIT_MB", "2400"))
    devices = tf.config.list_physical_devices("GPU")
    if not devices:
        return
    tf.config.set_logical_device_configuration(
        devices[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=limit_mb)],
    )


_configure_memory()

from advanced_visualization.tensorflow_service.app import (  # noqa: E402
    _cam,
    _custom_objects,
    _jet,
    _prepare_image,
)


def _valid_image(raw: str) -> Path | None:
    path = Path(str(raw)).expanduser()
    return path if path.is_file() else None


def _digest(path: Path) -> str:
    resolved = path.resolve()
    stat = resolved.stat()
    stamp = f"{resolved}:{stat.st_mtime_ns}:{stat.st_size}"
    return hashlib.sha1(stamp.encode("utf-8")).hexdigest()[:18]


def _output_path(artifact_dir: Path, layer: str, target: str, image: Path) -> Path:
    return (
        artifact_dir
        / "gradcam"
        / layer
        / target
        / f"{_digest(image)}_gradcampp_logit.webp"
    )


def _base_image(image: Image.Image, max_long_edge: int) -> np.ndarray:
    rendered = image.copy()
    if max(rendered.size) > max_long_edge:
        rendered.thumbnail((max_long_edge, max_long_edge), Image.Resampling.LANCZOS)
    return np.asarray(rendered, dtype=np.uint8)


def _save_overlay(base: np.ndarray, cam: tf.Tensor, path: Path) -> None:
    resized = tf.image.resize(cam[..., None], base.shape[:2], method="bilinear")
    values = tf.squeeze(resized).numpy()
    values = (values - values.min()) / (values.max() - values.min() + 1e-8)
    overlay = base.astype(np.float32) * 0.55 + _jet(values) * 0.45
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8), "RGB").save(
            temporary,
            format="WEBP",
            quality=WEBP_QUALITY,
            method=4,
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _logit_model(checkpoint: Path, layer_name: str, model_id: str) -> tf.keras.Model:
    model = tf.keras.models.load_model(
        str(checkpoint), custom_objects=_custom_objects(), compile=False
    )
    pred = model.get_layer("pred")
    logit_layer = tf.keras.layers.Dense(
        pred.units,
        activation=None,
        use_bias=pred.use_bias,
        name=f"{model_id}_fraud_logit",
    )
    logit = logit_layer(pred.input)
    logit_layer.set_weights(pred.get_weights())
    activation = model.get_layer(layer_name).output
    return tf.keras.Model(model.inputs, [activation, logit], name=model_id)


def _image_values(csv_path: Path, image_column: str):
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        try:
            image_index = header.index(image_column)
        except ValueError as exc:
            raise ValueError(
                f"Image column {image_column!r} is missing from {csv_path}."
            ) from exc
        for row in reader:
            if image_index < len(row):
                yield row[image_index]


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def generate(
    model_id: str,
    layer_key: str,
    limit: int | None,
    *,
    overwrite: bool = False,
    num_shards: int = 1,
    shard_index: int = 0,
) -> dict:
    if num_shards < 1:
        raise ValueError("num_shards must be at least 1.")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("shard_index must be in [0, num_shards).")
    route = registered_model_routes()[model_id]
    if route.framework != "tensorflow" or route.checkpoint is None:
        raise ValueError(f"{model_id} is not a checkpoint-backed TensorFlow route.")
    layer = route.layer(layer_key)
    image_column = route.columns["image"]
    artifact_dir = route.artifact_dir
    (artifact_dir / "features").mkdir(parents=True, exist_ok=True)
    explained = _logit_model(route.checkpoint, layer.module, model_id)

    image_paths: list[Path] = []
    seen: set[str] = set()
    invalid_sources = 0
    for raw_path in _image_values(route.prediction_data, image_column):
        image_path = _valid_image(raw_path)
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

    generated = skipped = failed = processed = 0
    state_path = artifact_dir / (
        f"gradcam_generation_state.{layer.key}."
        f"shard-{shard_index:03d}-of-{num_shards:03d}.json"
    )
    for image_path in image_paths:
        processed += 1
        outputs = {
            target: _output_path(artifact_dir, layer.key, target, image_path)
            for target in TARGETS
        }
        missing = [
            target
            for target, path in outputs.items()
            if overwrite or not path.is_file()
        ]
        if not missing:
            skipped += len(TARGETS)
            continue
        try:
            with Image.open(image_path) as opened:
                transposed = ImageOps.exif_transpose(opened)
                image = (transposed or opened).convert("RGB")
                base = _base_image(image, layer.max_long_edge)
                input_tensor = _prepare_image(image, route.image_size)
            with tf.GradientTape() as tape:
                activation, logit = explained(input_tensor, training=False)
                score = tf.reduce_mean(logit)
            gradient = tape.gradient(score, activation)
            if gradient is None:
                raise RuntimeError(f"Layer {layer.module} is disconnected from the logit.")
            cams = {
                "fraud": _cam(activation, gradient, "gradcam++"),
                "genuine": _cam(activation, -gradient, "gradcam++"),
            }
            for target in missing:
                _save_overlay(base, cams[target], outputs[target])
                generated += 1
        except Exception as exc:
            failed += len(missing)
            print(
                f"{model_id}:{layer.key}: {image_path}: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
        if processed % 100 == 0:
            state = {
                "model_id": model_id,
                "layer": layer.key,
                "num_shards": num_shards,
                "shard_index": shard_index,
                "processed": processed,
                "generated": generated,
                "skipped": skipped,
                "failed": failed,
                "invalid_sources": invalid_sources,
            }
            _atomic_json(state_path, state)
            print(json.dumps(state, sort_keys=True), flush=True)

    result = {
        "model_id": model_id,
        "layer": layer.key,
        "num_shards": num_shards,
        "shard_index": shard_index,
        "source_images": len(image_paths),
        "processed": processed,
        "generated": generated,
        "skipped": skipped,
        "failed": failed,
        "invalid_sources": invalid_sources,
        "complete": True,
    }
    _atomic_json(state_path, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_id")
    parser.add_argument("--layer", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    print(
        json.dumps(
            generate(
                arguments.model_id,
                arguments.layer,
                arguments.limit,
                overwrite=arguments.overwrite,
                num_shards=arguments.num_shards,
                shard_index=arguments.shard_index,
            )
        )
    )
