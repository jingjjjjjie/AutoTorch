"""In-memory prediction and class-targeted Grad-CAM for uploaded images."""

from __future__ import annotations

import base64
import io
import json
import os
import threading
import time
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps, UnidentifiedImageError

from advanced_visualization.core.config import ModelRunConfig
from advanced_visualization.core.heatmap import jet_overlay
from advanced_visualization.models.gradcam import engine_for_config


MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
MAX_RENDER_SIDE = 1600


@dataclass(frozen=True)
class LiveInferenceResult:
    fraud_probability: float
    genuine_probability: float
    logit: float
    predicted_class: str
    threshold: float
    fraud_gradcam: str
    genuine_gradcam: str
    width: int
    height: int
    elapsed_ms: int

    def to_dict(self) -> dict[str, object]:
        return self.__dict__.copy()


def decode_uploaded_image(payload: bytes) -> Image.Image:
    """Validate an uploaded payload and return an orientation-correct RGB image."""
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


def _data_url(image: Image.Image) -> str:
    output = io.BytesIO()
    image.save(output, format="PNG", compress_level=3)
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


class LiveInferenceService:
    """Serialize access to cached models while hooks and autograd are active."""

    def __init__(self) -> None:
        self._lock = threading.Lock()

    def infer(
        self,
        config: ModelRunConfig,
        payload: bytes,
        *,
        threshold: float = 0.5,
        method: str = "gradcam",
    ) -> LiveInferenceResult:
        if method not in {"gradcam", "gradcam++"}:
            raise ValueError("CAM method must be gradcam or gradcam++.")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Prediction threshold must be between 0 and 1.")

        image = decode_uploaded_image(payload)
        started = time.perf_counter()
        with self._lock:
            engine = engine_for_config(config)
            bundle = engine.load_bundle(config)
            activations: dict[str, torch.Tensor] = {}

            def capture_activation(_module, _inputs, output) -> None:
                activations["value"] = output

            handle = bundle.target_layer.register_forward_hook(capture_activation)
            try:
                input_tensor = bundle.transform(image).unsqueeze(0).to(bundle.device)
                bundle.model.zero_grad(set_to_none=True)
                logit = engine.score(bundle.model, input_tensor).mean()
                activation = activations.get("value")
                if activation is None or not isinstance(activation, torch.Tensor):
                    raise RuntimeError("The configured Grad-CAM layer produced no tensor activation.")
                gradient = torch.autograd.grad(logit, activation, retain_graph=False)[0]
                fraud_cam = engine.compute_cam(
                    activation.detach(), gradient.detach(), method=method
                ).cpu()
                genuine_cam = engine.compute_cam(
                    activation.detach(), -gradient.detach(), method=method
                ).cpu()
                fraud_probability = float(torch.sigmoid(logit.detach()).cpu())
                raw_logit = float(logit.detach().cpu())
            finally:
                handle.remove()

        render_image = image.copy()
        render_image.thumbnail((MAX_RENDER_SIDE, MAX_RENDER_SIDE), Image.Resampling.LANCZOS)
        target_size = (render_image.height, render_image.width)

        def overlay(cam: torch.Tensor) -> str:
            resized = F.interpolate(
                cam, size=target_size, mode="bilinear", align_corners=False
            ).squeeze().float()
            resized = (resized - resized.min()) / (resized.max() - resized.min() + 1e-8)
            rendered = jet_overlay(
                np.asarray(render_image), resized.detach().cpu().numpy()
            )
            return _data_url(rendered)

        elapsed_ms = round((time.perf_counter() - started) * 1000)
        return LiveInferenceResult(
            fraud_probability=fraud_probability,
            genuine_probability=1.0 - fraud_probability,
            logit=raw_logit,
            predicted_class="fraud" if fraud_probability >= threshold else "genuine",
            threshold=threshold,
            fraud_gradcam=overlay(fraud_cam),
            genuine_gradcam=overlay(genuine_cam),
            width=image.width,
            height=image.height,
            elapsed_ms=elapsed_ms,
        )


live_inference_service = LiveInferenceService()


class RemoteInferenceError(RuntimeError):
    """A remote inference service was unavailable or rejected the request."""

    def __init__(self, message: str, status_code: int = 503) -> None:
        super().__init__(message)
        self.status_code = status_code


class TensorFlowLiveInferenceClient:
    """Small HTTP client for the isolated TensorFlow 2.8 VAN Small service."""

    def __init__(self, base_url: str | None = None, timeout_seconds: int = 300) -> None:
        self.base_url = (base_url if base_url is not None else os.environ.get("TF_VAN_LIVE_URL", "")).rstrip("/")
        self.timeout_seconds = timeout_seconds

    @property
    def configured(self) -> bool:
        return bool(self.base_url)

    def _request(self, path: str, *, payload: bytes | None = None) -> object:
        if not self.configured:
            raise RemoteInferenceError("TensorFlow VAN Small live inference is not configured.")
        request = Request(
            self.base_url + path,
            data=payload,
            headers={"Content-Type": "application/octet-stream"} if payload is not None else {},
            method="POST" if payload is not None else "GET",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            try:
                detail = json.loads(exc.read().decode("utf-8")).get("detail")
            except (ValueError, AttributeError):
                detail = None
            raise RemoteInferenceError(detail or f"TensorFlow inference failed ({exc.code}).", exc.code) from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise RemoteInferenceError(f"TensorFlow VAN Small service is unavailable: {exc}") from exc

    def models(self) -> list[dict[str, object]]:
        if not self.configured:
            return []
        result = self._request("/models")
        if not isinstance(result, list):
            raise RemoteInferenceError("TensorFlow VAN Small service returned an invalid model list.")
        return [item for item in result if isinstance(item, dict)]

    def predict(
        self,
        model_key: str,
        payload: bytes,
        *,
        threshold: float,
        method: str,
    ) -> dict[str, object]:
        query = urlencode(
            {"model_key": model_key, "threshold": threshold, "method": method}
        )
        result = self._request(f"/predict?{query}", payload=payload)
        if not isinstance(result, dict):
            raise RemoteInferenceError("TensorFlow VAN Small service returned an invalid result.")
        return result


tensorflow_live_inference_client = TensorFlowLiveInferenceClient()
