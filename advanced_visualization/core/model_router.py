"""Load model-local ``router.py`` files into one validated application contract."""

from __future__ import annotations

import hashlib
import importlib.util
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping


ROUTER_FILENAME = "router.py"
ROUTER_API_VERSION = 1
DEFAULT_ARTIFACT_ROOT = Path(
    os.environ.get(
        "ADVANCED_VISUALIZATION_ARTIFACT_ROOT",
        "/mnt4/advanced_visualization",
    )
)
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class GradcamLayerRoute:
    """One selectable layer exposed by a model router."""

    key: str
    module: str
    label: str
    final: bool = False
    max_long_edge: int = 768


@dataclass(frozen=True)
class ModelRoute:
    """Normalized data/model locations returned by a model-local router."""

    model_id: str
    data_dir: Path
    router_path: Path
    artifact_dir: Path
    prediction_data: Path
    feature_data: Path | None
    checkpoint: Path | None
    framework: str
    engine: str
    model_name: str
    head_type: str
    image_size: int
    columns: dict[str, str]
    layers: tuple[GradcamLayerRoute, ...]
    prepared_gradcam_layers: tuple[str, ...] | None = None
    gradcam_montage_column: str = ""
    gradcam_montage_layers: tuple[str, ...] = ()
    branch: str = ""
    review_preset: dict[str, Any] = field(default_factory=dict)

    @property
    def final_layers(self) -> tuple[GradcamLayerRoute, ...]:
        return tuple(layer for layer in self.layers if layer.final)

    @property
    def non_final_layers(self) -> tuple[GradcamLayerRoute, ...]:
        return tuple(layer for layer in self.layers if not layer.final)

    def layer(self, key: str) -> GradcamLayerRoute:
        for layer in self.layers:
            if layer.key == key:
                return layer
        available = ", ".join(layer.key for layer in self.layers)
        raise KeyError(f"Unknown layer {key!r} for {self.model_id}. Available: {available}")


def _path(value: Any, data_dir: Path, *, required: bool = False) -> Path | None:
    raw = str(value or "").strip()
    if not raw:
        if required:
            raise ValueError("Router did not provide a required path.")
        return None
    path = Path(raw).expanduser()
    return path if path.is_absolute() else data_dir / path


def _load_module(router_path: Path) -> ModuleType:
    stat = router_path.stat()
    return _load_module_cached(
        str(router_path.resolve()), int(stat.st_mtime_ns), int(stat.st_size)
    )


@lru_cache(maxsize=64)
def _load_module_cached(
    router_path: str, _mtime_ns: int, _size: int
) -> ModuleType:
    digest = hashlib.sha1(router_path.encode("utf-8")).hexdigest()[:16]
    module_name = f"advanced_visualization_data_router_{digest}"
    spec = importlib.util.spec_from_file_location(module_name, router_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import model router: {router_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _layer_routes(raw_layers: Any) -> tuple[GradcamLayerRoute, ...]:
    layers: list[GradcamLayerRoute] = []
    for raw in raw_layers or ():
        if not isinstance(raw, Mapping):
            raise TypeError("Each Grad-CAM layer route must be a mapping.")
        key = str(raw.get("key") or "").strip()
        module = str(raw.get("module") or raw.get("name") or "").strip()
        if not key or not SAFE_ID.fullmatch(key):
            raise ValueError(f"Invalid router layer key: {key!r}")
        if not module:
            raise ValueError(f"Layer {key!r} has no module path/name.")
        final = bool(raw.get("final", False))
        max_long_edge = int(raw.get("max_long_edge") or (1024 if final else 768))
        if max_long_edge < 1:
            raise ValueError(f"Layer {key!r} has an invalid max_long_edge.")
        layers.append(
            GradcamLayerRoute(
                key=key,
                module=module,
                label=str(raw.get("label") or key),
                final=final,
                max_long_edge=max_long_edge,
            )
        )
    keys = [layer.key for layer in layers]
    if len(keys) != len(set(keys)):
        raise ValueError("Router Grad-CAM layer keys must be unique.")
    return tuple(layers)


def load_model_route(model_id: str, data_dir: Path | str) -> ModelRoute:
    """Execute ``data_dir/router.py`` and normalize its returned route."""

    model_id = str(model_id).strip()
    if not SAFE_ID.fullmatch(model_id):
        raise ValueError(f"Invalid model_id: {model_id!r}")
    directory = Path(data_dir).expanduser()
    if not directory.exists():
        from advanced_visualization.core.settings import configured_path

        directory = configured_path(str(directory))
    router_path = directory / ROUTER_FILENAME
    if not router_path.is_file():
        raise FileNotFoundError(f"Model router does not exist: {router_path}")
    module = _load_module(router_path)
    api_version = int(getattr(module, "ROUTER_API_VERSION", ROUTER_API_VERSION))
    if api_version != ROUTER_API_VERSION:
        raise ValueError(
            f"Unsupported router API version {api_version} in {router_path}; "
            f"expected {ROUTER_API_VERSION}."
        )
    resolver = getattr(module, "resolve", None)
    if not callable(resolver):
        raise AttributeError(f"{router_path} must define resolve(model_id, data_dir).")
    raw = resolver(model_id=model_id, data_dir=directory)
    if not isinstance(raw, Mapping):
        raise TypeError(f"{router_path} resolve() must return a mapping.")

    columns = {
        str(key): str(value)
        for key, value in dict(raw.get("columns") or {}).items()
        if str(value).strip()
    }
    prediction_data = _path(
        raw.get("prediction_data") or raw.get("prediction_csv"),
        directory,
        required=True,
    )
    checkpoint = _path(raw.get("checkpoint"), directory)
    layers = _layer_routes(raw.get("layers"))
    layer_keys = {layer.key for layer in layers}
    raw_prepared_layers = raw.get("prepared_gradcam_layers")
    prepared_gradcam_layers = (
        tuple(str(item) for item in raw_prepared_layers)
        if raw_prepared_layers is not None
        else None
    )
    if prepared_gradcam_layers is not None:
        unknown = set(prepared_gradcam_layers) - layer_keys
        if unknown:
            raise ValueError(
                f"Unknown prepared Grad-CAM layers in {router_path}: "
                f"{', '.join(sorted(unknown))}"
            )
    gradcam_montage_layers = tuple(
        str(item) for item in raw.get("gradcam_montage_layers") or ()
    )
    unknown_montage_layers = set(gradcam_montage_layers) - layer_keys
    if unknown_montage_layers:
        raise ValueError(
            f"Unknown montage layers in {router_path}: "
            f"{', '.join(sorted(unknown_montage_layers))}"
        )
    framework = str(raw.get("framework") or "artifact").strip().lower()
    artifact_root = Path(
        os.environ.get(
            "ADVANCED_VISUALIZATION_ARTIFACT_ROOT",
            str(DEFAULT_ARTIFACT_ROOT),
        )
    ).expanduser()
    routed_artifact_dir = _path(raw.get("artifact_dir"), directory)
    return ModelRoute(
        model_id=model_id,
        data_dir=directory,
        router_path=router_path,
        artifact_dir=routed_artifact_dir or artifact_root / model_id,
        prediction_data=prediction_data,
        feature_data=_path(raw.get("feature_data") or raw.get("feature_csv"), directory),
        checkpoint=checkpoint,
        framework=framework,
        engine=str(raw.get("engine") or raw.get("model_type") or framework),
        model_name=str(raw.get("model_name") or ""),
        head_type=str(raw.get("head_type") or ""),
        image_size=int(raw.get("image_size") or 0),
        columns=columns,
        layers=layers,
        prepared_gradcam_layers=prepared_gradcam_layers,
        gradcam_montage_column=str(
            raw.get("gradcam_montage_column") or ""
        ).strip(),
        gradcam_montage_layers=gradcam_montage_layers,
        branch=str(raw.get("branch") or ""),
        review_preset=dict(raw.get("review_preset") or {}),
    )


def registered_model_routes() -> dict[str, ModelRoute]:
    """Load enabled router-backed registrations from user settings."""

    from advanced_visualization.core.settings import load_settings

    routes: dict[str, ModelRoute] = {}
    for model in load_settings().models:
        if not model.enabled or not model.key.strip() or not model.data_dir.strip():
            continue
        try:
            routes[model.key] = load_model_route(model.key, model.data_dir)
        except FileNotFoundError:
            # A registered output workspace may be intentionally absent while it
            # is being rebuilt. Other complete model registrations remain usable.
            continue
    return routes


def route_for_model(model_id: str) -> ModelRoute | None:
    return registered_model_routes().get(model_id)
