"""TensorFlow 2.8 VAN Small prediction and multi-layer Grad-CAM service."""

from __future__ import annotations

import base64
import io
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf
from fastapi import Body, FastAPI, HTTPException, Query
from PIL import Image, ImageOps, UnidentifiedImageError


MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
DEFAULT_IMAGE_SIZE = 512


def _configure_gpu_memory() -> None:
    limit_mb = int(os.environ.get("VANSMALL_GPU_MEMORY_LIMIT_MB", "2400"))
    devices = tf.config.list_physical_devices("GPU")
    if devices and not tf.config.get_logical_device_configuration(devices[0]):
        tf.config.set_logical_device_configuration(
            devices[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=limit_mb)],
        )


_configure_gpu_memory()


@dataclass(frozen=True)
class BranchConfig:
    key: str
    label: str
    checkpoint: Path
    image_size: int = DEFAULT_IMAGE_SIZE
    layers: Tuple[Tuple[str, str, str], ...] = ()


def _branch_configs() -> Dict[str, BranchConfig]:
    from advanced_visualization.core.model_router import registered_model_routes

    configs = {}
    for route in registered_model_routes().values():
        if route.framework != "tensorflow" or route.checkpoint is None:
            continue
        configs[route.model_id] = BranchConfig(
            key=route.model_id,
            label=f"{route.model_name} · {route.branch or route.model_id}",
            checkpoint=route.checkpoint,
            image_size=route.image_size or DEFAULT_IMAGE_SIZE,
            layers=tuple(
                (layer.module, layer.label, layer.key) for layer in route.layers
            ),
        )
    return configs


def _custom_objects() -> dict:
    # The mounted IDRecapture repository owns these serialized Keras layers.
    from tfvan.block import Block
    from tfvan.embed import PatchEmbedding
    from tfvan.norm import LayerNorm

    return {
        "PatchEmbedding": PatchEmbedding,
        "Block": Block,
        "LayerNorm": LayerNorm,
    }


def _decode_image(payload: bytes) -> Image.Image:
    if not payload:
        raise ValueError("Choose an image before running inference.")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise ValueError("Image is larger than the 20 MB upload limit.")
    try:
        with Image.open(io.BytesIO(payload)) as uploaded:
            width, height = uploaded.size
            if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
                raise ValueError("Image dimensions are invalid or exceed 40 megapixels.")
            uploaded.verify()
        with Image.open(io.BytesIO(payload)) as uploaded:
            return ImageOps.exif_transpose(uploaded).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("The upload is not a readable image.") from exc


def _prepare_image(image: Image.Image, image_size: int) -> tf.Tensor:
    value = tf.convert_to_tensor(np.asarray(image), dtype=tf.float32)
    height = tf.shape(value)[0]
    width = tf.shape(value)[1]
    if int(height.numpy()) >= int(width.numpy()):
        size = tf.minimum(height, width)
        top = (height - size) // 2
        left = (width - size) // 2
        value = tf.image.crop_to_bounding_box(value, top, left, size, size)
    value = tf.image.resize_with_pad(value, image_size, image_size)
    return tf.expand_dims(value, axis=0)


def _png_data_url(image: np.ndarray) -> str:
    output = io.BytesIO()
    Image.fromarray(np.clip(image, 0, 255).astype(np.uint8), "RGB").save(
        output, format="PNG", compress_level=3
    )
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return "data:image/png;base64," + encoded


def _jet(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, 0.0, 1.0)
    red = np.clip(1.5 - np.abs(4.0 * values - 3.0), 0.0, 1.0)
    green = np.clip(1.5 - np.abs(4.0 * values - 2.0), 0.0, 1.0)
    blue = np.clip(1.5 - np.abs(4.0 * values - 1.0), 0.0, 1.0)
    return np.stack((red, green, blue), axis=-1) * 255.0


def _overlay(image: np.ndarray, cam: tf.Tensor) -> str:
    resized = tf.image.resize(cam[..., None], image.shape[:2], method="bilinear")
    values = tf.squeeze(resized).numpy()
    minimum = float(values.min())
    maximum = float(values.max())
    values = (values - minimum) / (maximum - minimum + 1e-8)
    rendered = image.astype(np.float32) * 0.55 + _jet(values) * 0.45
    return _png_data_url(rendered)


def _cam(activation: tf.Tensor, gradient: tf.Tensor, method: str) -> tf.Tensor:
    if method == "gradcam++":
        second = gradient * gradient
        third = second * gradient
        activation_sum = tf.reduce_sum(activation, axis=(1, 2), keepdims=True)
        denominator = 2.0 * second + third * activation_sum
        denominator = tf.where(tf.abs(denominator) > 1e-10, denominator, tf.ones_like(denominator))
        alpha = second / denominator
        weights = tf.reduce_sum(alpha * tf.nn.relu(gradient), axis=(1, 2), keepdims=True)
    else:
        weights = tf.reduce_mean(gradient, axis=(1, 2), keepdims=True)
    return tf.nn.relu(tf.reduce_sum(weights * activation, axis=-1))[0]


class VanSmallService:
    def __init__(self) -> None:
        self._models: Dict[str, tf.keras.Model] = {}
        self._lock = threading.RLock()

    def _load(self, config: BranchConfig) -> tf.keras.Model:
        with self._lock:
            cached = self._models.get(config.key)
            if cached is not None:
                return cached
            if not config.checkpoint.is_file():
                raise FileNotFoundError(str(config.checkpoint))
            model = tf.keras.models.load_model(
                str(config.checkpoint), custom_objects=_custom_objects(), compile=False
            )
            pred = model.get_layer("pred")
            logit_layer = tf.keras.layers.Dense(
                pred.units,
                activation=None,
                use_bias=pred.use_bias,
                name="fraud_logit",
            )
            logit = logit_layer(pred.input)
            logit_layer.set_weights(pred.get_weights())
            outputs = [
                model.get_layer(name).output for name, _label, _key in config.layers
            ]
            explained = tf.keras.Model(model.inputs, outputs + [logit], name=config.key)
            self._models[config.key] = explained
            return explained

    def infer(
        self,
        config: BranchConfig,
        payload: bytes,
        threshold: float,
        method: str,
    ) -> dict:
        image = _decode_image(payload)
        input_tensor = _prepare_image(image, config.image_size)
        display_image = np.clip(input_tensor[0].numpy(), 0, 255).astype(np.uint8)
        started = time.perf_counter()
        with self._lock:
            model = self._load(config)
            with tf.GradientTape(persistent=True) as tape:
                values: List[tf.Tensor] = model(input_tensor, training=False)
                activations = values[:-1]
                logit = tf.reduce_mean(values[-1])
            gradients = [tape.gradient(logit, activation) for activation in activations]
            del tape
        if any(gradient is None for gradient in gradients):
            raise RuntimeError("A configured VAN Small layer is disconnected from the fraud logit.")
        raw_logit = float(logit.numpy())
        fraud_probability = float(tf.math.sigmoid(logit).numpy())
        gradcams = []
        for (layer_name, layer_label, layer_key), activation, gradient in zip(
            config.layers, activations, gradients
        ):
            fraud_cam = _cam(activation, gradient, method)
            genuine_cam = _cam(activation, -gradient, method)
            gradcams.append(
                {
                    "layer": layer_key,
                    "label": layer_label,
                    "fraud_gradcam": _overlay(display_image, fraud_cam),
                    "genuine_gradcam": _overlay(display_image, genuine_cam),
                }
            )
        elapsed_ms = round((time.perf_counter() - started) * 1000)
        return {
            "model_key": config.key,
            "branch": "crop" if config.key.endswith("_crop") else "ori",
            "fraud_probability": fraud_probability,
            "genuine_probability": 1.0 - fraud_probability,
            "logit": raw_logit,
            "predicted_class": "fraud" if fraud_probability >= threshold else "genuine",
            "threshold": threshold,
            "input_image": _png_data_url(display_image),
            "fraud_gradcam": gradcams[0]["fraud_gradcam"],
            "genuine_gradcam": gradcams[0]["genuine_gradcam"],
            "gradcams": gradcams,
            "width": image.width,
            "height": image.height,
            "image_size": config.image_size,
            "elapsed_ms": elapsed_ms,
        }


service = VanSmallService()
app = FastAPI(title="Ench21 VAN Small live inference", version="1.0.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/models")
def models() -> list:
    return [
        {
            "key": config.key,
            "label": config.label,
            "model_name": "vansmall",
            "framework": "tensorflow",
            "branch": "crop" if config.key.endswith("_crop") else "ori",
            "image_size": config.image_size,
            "available": config.checkpoint.is_file(),
            "threshold": 0.5,
            "layers": [label for _name, label, _key in config.layers],
        }
        for config in _branch_configs().values()
    ]


@app.post("/predict")
def predict(
    model_key: str = Query(...),
    threshold: float = Query(0.5, ge=0.0, le=1.0),
    method: str = Query("gradcam++", regex="^gradcam\\+\\+$"),
    payload: bytes = Body(..., media_type="application/octet-stream"),
) -> dict:
    config = _branch_configs().get(model_key)
    if config is None:
        raise HTTPException(status_code=404, detail="Unknown TensorFlow VAN Small model.")
    if not config.checkpoint.is_file():
        raise HTTPException(status_code=409, detail="The selected TensorFlow checkpoint is unavailable.")
    try:
        return service.infer(config, payload, threshold, method)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (FileNotFoundError, RuntimeError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
