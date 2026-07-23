"""Run and artifact configuration for visualization tools.

Keep model/run differences here. The Streamlit app should consume the
standardized manifest produced by preparation.py and avoid hardcoding model
paths directly.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from advanced_visualization.core.settings import configured_path, load_settings, require_setting


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
_SETTINGS = load_settings()

DEFAULT_GRADCAM_ROOT = configured_path(require_setting(_SETTINGS.default_gradcam_root, "default_gradcam_root"))
MANIFEST_NAME = require_setting(_SETTINGS.manifest_name, "manifest_name")
PREPARED_CSV_NAME = require_setting(_SETTINGS.prepared_csv_name, "prepared_csv_name")

IMAGE_COLUMNS = tuple(require_setting(_SETTINGS.image_columns, "image_columns"))
ID_COLUMNS = tuple(require_setting(_SETTINGS.id_columns, "id_columns"))
SUBCLASS_COLUMNS = tuple(require_setting(_SETTINGS.subclass_columns, "subclass_columns"))
IMAGE_EXTENSIONS = set(require_setting(_SETTINGS.image_extensions, "image_extensions"))
DEFAULT_MEAN = tuple(require_setting(_SETTINGS.normalize_mean, "normalize_mean"))
DEFAULT_STD = tuple(require_setting(_SETTINGS.normalize_std, "normalize_std"))


@dataclass(frozen=True)
class ModelRunConfig:
    """Everything that differs between model runs."""

    key: str
    checkpoint: Path
    model_name: str
    head_type: str
    image_size: int
    image_column: str = ""
    gradcam_engine: str = "unireplknet"
    transform_version: str = "v1"
    prediction_column: str = ""

    def to_json_dict(self) -> dict:
        payload = asdict(self)
        payload["checkpoint"] = str(self.checkpoint)
        return payload


def configured_model_runs() -> dict[str, ModelRunConfig]:
    runs: dict[str, ModelRunConfig] = {}
    for model in load_settings().models:
        if model.enabled and model.key.strip() and model.data_dir.strip():
            from advanced_visualization.core.model_router import load_model_route

            route = load_model_route(model.key, model.data_dir)
            if route.framework != "pytorch" or route.checkpoint is None:
                continue
            runs[model.key] = ModelRunConfig(
                key=model.key,
                checkpoint=route.checkpoint,
                model_name=route.model_name,
                head_type=route.head_type,
                image_size=route.image_size,
                image_column=route.columns.get("image", ""),
                gradcam_engine=route.engine,
                prediction_column=route.columns.get("prediction", ""),
            )
            continue
        checkpoint = model.resolved_checkpoint()
        if not model.enabled or not model.key.strip() or checkpoint is None:
            continue
        runs[model.key] = ModelRunConfig(
            key=model.key,
            checkpoint=checkpoint,
            model_name=model.model_name,
            head_type=model.head_type,
            image_size=model.image_size,
            image_column=model.image_column,
            gradcam_engine=model.model_type,
            prediction_column=model.prediction_column,
        )
    return runs


MODEL_RUNS: dict[str, ModelRunConfig] = configured_model_runs()


def all_model_runs() -> dict[str, ModelRunConfig]:
    return configured_model_runs()


def run_key_from_checkpoint(checkpoint: Path) -> Optional[str]:
    checkpoint = checkpoint.expanduser()
    for key, config in all_model_runs().items():
        if config.checkpoint.expanduser() == checkpoint:
            return key
    return None


def run_key_from_epoch(artifact_dir: Path, epoch: int) -> Optional[str]:
    checkpoint = artifact_dir.expanduser() / "checkpoints" / f"epoch_{epoch}.pt"
    return run_key_from_checkpoint(checkpoint)


def model_run(key: str) -> ModelRunConfig:
    runs = all_model_runs()
    if key not in runs:
        raise KeyError(f"Unknown model run config: {key}")
    return runs[key]


def gradcam_artifact_root(config_key: str) -> Optional[Path]:
    from advanced_visualization.core.model_router import route_for_model

    route = route_for_model(config_key)
    if route is not None:
        return route.artifact_dir / "gradcam"
    config = all_model_runs().get(config_key)
    if not config:
        return None
    checkpoint = config.checkpoint
    if checkpoint.parent.name == "checkpoints":
        return checkpoint.parent.parent / "gradcam"
    return checkpoint.parent / "gradcam"
