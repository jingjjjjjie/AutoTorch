"""Tests for uploaded-image prediction and Grad-CAM generation."""

from __future__ import annotations

import io
from pathlib import Path

import pytest
import torch
from PIL import Image

from advanced_visualization.core.config import ModelRunConfig
from advanced_visualization.models.base import GradcamBundle
from advanced_visualization.web import live_inference


def image_payload() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (18, 12), (120, 80, 40)).save(output, format="PNG")
    return output.getvalue()


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.target = torch.nn.Conv2d(3, 2, kernel_size=1, bias=False)
        torch.nn.init.constant_(self.target.weight, 0.25)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.target(value).mean()


class TinyEngine:
    def __init__(self) -> None:
        self.model = TinyModel()

    def load_bundle(self, _config: ModelRunConfig) -> GradcamBundle:
        def transform(image: Image.Image) -> torch.Tensor:
            values = torch.tensor(list(image.resize((8, 8)).getdata()), dtype=torch.float32)
            return values.reshape(8, 8, 3).permute(2, 0, 1) / 255.0

        return GradcamBundle(self.model, transform, self.model.target, torch.device("cpu"))

    def score(self, model: TinyModel, input_tensor: torch.Tensor) -> torch.Tensor:
        return model(input_tensor)

    def compute_cam(
        self, activation: torch.Tensor, gradient: torch.Tensor, method: str = "gradcam"
    ) -> torch.Tensor:
        del method
        weights = gradient.mean(dim=(2, 3), keepdim=True)
        return torch.relu((weights * activation).sum(dim=1, keepdim=True))


def test_live_inference_returns_prediction_and_both_gradcams(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(live_inference, "engine_for_config", lambda _config: TinyEngine())
    config = ModelRunConfig(
        key="tiny",
        checkpoint=Path("unused.pt"),
        model_name="tiny",
        head_type="tiny",
        image_size=8,
    )

    result = live_inference.LiveInferenceService().infer(
        config, image_payload(), threshold=0.5
    )

    assert result.predicted_class == "fraud"
    assert result.fraud_probability + result.genuine_probability == pytest.approx(1.0)
    assert result.fraud_gradcam.startswith("data:image/png;base64,")
    assert result.genuine_gradcam.startswith("data:image/png;base64,")
    assert (result.width, result.height) == (18, 12)


def test_uploaded_image_validation_rejects_non_image() -> None:
    with pytest.raises(ValueError, match="not a readable image"):
        live_inference.decode_uploaded_image(b"not an image")


def test_tensorflow_client_is_optional_when_url_is_not_configured() -> None:
    client = live_inference.TensorFlowLiveInferenceClient(base_url="")

    assert client.configured is False
    assert client.models() == []


def test_tensorflow_client_forwards_model_threshold_and_method(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = live_inference.TensorFlowLiveInferenceClient(base_url="http://tensorflow")
    captured = {}

    def fake_request(path: str, *, payload: bytes | None = None) -> object:
        captured.update(path=path, payload=payload)
        return {"predicted_class": "genuine", "gradcams": []}

    monkeypatch.setattr(client, "_request", fake_request)
    result = client.predict(
        "ench21_vansmall_ori",
        b"image",
        threshold=0.25,
        method="gradcam++",
    )

    assert result["predicted_class"] == "genuine"
    assert captured["payload"] == b"image"
    assert "model_key=ench21_vansmall_ori" in captured["path"]
    assert "threshold=0.25" in captured["path"]
    assert "method=gradcam%2B%2B" in captured["path"]
