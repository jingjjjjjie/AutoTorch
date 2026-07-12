"""Persistent user settings for visualization paths and model runs."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SETTINGS_PATH = PACKAGE_ROOT / "settings.json"


@dataclass
class UserModelConfig:
    key: str = ""
    prediction_csv: str = ""
    feature_csv: str = ""
    artifact_dir: str = ""
    checkpoint: str = ""
    weights_epoch: int | None = None
    model_type: str = ""
    model_name: str = ""
    head_type: str = ""
    image_size: int = 0
    image_column: str = ""
    prediction_column: str = ""
    enabled: bool = True

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "UserModelConfig":
        weights_epoch = payload.get("weights_epoch")
        return cls(
            key=str(payload.get("key", "")),
            prediction_csv=str(payload.get("prediction_csv", "")),
            feature_csv=str(payload.get("feature_csv", "")),
            artifact_dir=str(payload.get("artifact_dir", "")),
            checkpoint=str(payload.get("checkpoint", "")),
            weights_epoch=int(weights_epoch) if str(weights_epoch or "").strip() else None,
            model_type=str(payload.get("model_type", payload.get("gradcam_engine", ""))),
            model_name=str(payload.get("model_name", "")),
            head_type=str(payload.get("head_type", "")),
            image_size=int(payload.get("image_size") or 0),
            image_column=str(payload.get("image_column", "")),
            prediction_column=str(payload.get("prediction_column", "")),
            enabled=bool(payload.get("enabled", True)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "prediction_csv": self.prediction_csv,
            "feature_csv": self.feature_csv,
            "artifact_dir": self.artifact_dir,
            "checkpoint": self.checkpoint,
            "weights_epoch": self.weights_epoch,
            "model_type": self.model_type,
            "model_name": self.model_name,
            "head_type": self.head_type,
            "image_size": self.image_size,
            "image_column": self.image_column,
            "prediction_column": self.prediction_column,
            "enabled": self.enabled,
        }

    def resolved_checkpoint(self) -> Path | None:
        if self.checkpoint.strip():
            return configured_path(self.checkpoint)
        if self.artifact_dir.strip() and self.weights_epoch is not None:
            return configured_path(self.artifact_dir) / "checkpoints" / f"epoch_{self.weights_epoch}.pt"
        return None


DEFAULT_EXTRA_VIEW_CONFIGS = [
    {
        "model_type": "vansmall",
        "label": "VAN Small Montage Review",
        "description": "Crop/Ori VAN Small 2x2 logit Grad-CAM montage review.",
        "view": "layered_gradcam",
        "score": "logit",
        "column_template": "tf_{branch}_{layer}_{score}_gradcam_path",
        "required_columns": [
            "tf_crop_layer_montage_path",
            "tf_ori_layer_montage_path",
        ],
        "branches": [
            {
                "key": "crop",
                "label": "Crop",
                "image_candidates": ["absolute_ocr_path", "ocr_path", "crop_path"],
            },
            {
                "key": "ori",
                "label": "Ori",
                "image_candidates": ["absolute_ori_path", "ori_path", "image_path", "path"],
            },
        ],
        "layers": [
            {
                "key": "montage",
                "label": "2x2 layer montage",
                "column_template": "tf_{branch}_layer_montage_path",
                "display": "single",
            },
            {"key": "norm3", "label": "norm3"},
            {"key": "block3_3", "label": "block3.3"},
            {"key": "block4_1", "label": "block4.1"},
            {"key": "norm4", "label": "norm4"},
        ],
        "default_layer": "montage",
        "prediction_candidates": ["tf_parallel_pred", "tf_crop_pred", "tf_ori_pred", "ypred_raw", "pred", "score"],
        "metadata_columns": ["Recapture_Subclass", "Data_Identity", "Quality_Issue", "fraud_type"],
    },
    {
        "model_type": "unireplknet",
        "label": "UniRepLKNet Grad-CAM Review",
        "description": "Configurable Grad-CAM artifact review for UniRepLKNet prepared outputs.",
        "view": "layered_gradcam",
        "score": "logit",
        "column_template": "unireplknet_{branch}_{layer}_{score}_gradcam_path",
        "required_columns": [],
        "branches": [
            {
                "key": "image",
                "label": "Image",
                "image_candidates": ["absolute_ori_path", "absolute_ocr_path", "image_path", "path"],
            }
        ],
        "layers": [
            {"key": "default", "label": "Prepared Grad-CAM", "column_candidates": ["gradcam_path", "gradcam", "heatmap_path"]},
            {"key": "stage4", "label": "stage4"},
        ],
        "prediction_candidates": ["prediction", "pred", "prob", "score", "result"],
        "metadata_columns": ["Recapture_Subclass", "Data_Identity", "Quality_Issue"],
    },
]


@dataclass
class UserSettings:
    prediction_csv: str = ""
    manifest_name: str = ""
    prepared_csv_name: str = ""
    default_gradcam_root: str = ""
    image_columns: list[str] = field(default_factory=list)
    id_columns: list[str] = field(default_factory=list)
    subclass_columns: list[str] = field(default_factory=list)
    image_extensions: list[str] = field(default_factory=list)
    normalize_mean: list[float] = field(default_factory=list)
    normalize_std: list[float] = field(default_factory=list)
    review: dict[str, Any] = field(default_factory=dict)
    model_type_options: list[str] = field(default_factory=list)
    image_size_options: list[int] = field(default_factory=list)
    extra_view_configs: list[dict[str, Any]] = field(default_factory=list)
    pipeline: dict[str, Any] = field(default_factory=dict)
    models: list[UserModelConfig] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "UserSettings":
        return cls(
            prediction_csv=str(payload.get("prediction_csv", "")),
            manifest_name=str(payload.get("manifest_name", "")),
            prepared_csv_name=str(payload.get("prepared_csv_name", "")),
            default_gradcam_root=str(payload.get("default_gradcam_root", "")),
            image_columns=[str(item) for item in payload.get("image_columns", [])],
            id_columns=[str(item) for item in payload.get("id_columns", [])],
            subclass_columns=[str(item) for item in payload.get("subclass_columns", [])],
            image_extensions=[str(item) for item in payload.get("image_extensions", [])],
            normalize_mean=[float(item) for item in payload.get("normalize_mean", [])],
            normalize_std=[float(item) for item in payload.get("normalize_std", [])],
            review=dict(payload.get("review", {})),
            model_type_options=[str(item) for item in payload.get("model_type_options", [])],
            image_size_options=[int(item) for item in payload.get("image_size_options", [])],
            extra_view_configs=list(payload.get("extra_view_configs", DEFAULT_EXTRA_VIEW_CONFIGS)),
            pipeline=dict(payload.get("pipeline", {})),
            models=[UserModelConfig.from_dict(item) for item in payload.get("models", []) if isinstance(item, dict)],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "prediction_csv": self.prediction_csv,
            "manifest_name": self.manifest_name,
            "prepared_csv_name": self.prepared_csv_name,
            "default_gradcam_root": self.default_gradcam_root,
            "image_columns": self.image_columns,
            "id_columns": self.id_columns,
            "subclass_columns": self.subclass_columns,
            "image_extensions": self.image_extensions,
            "normalize_mean": self.normalize_mean,
            "normalize_std": self.normalize_std,
            "review": self.review,
            "model_type_options": self.model_type_options,
            "image_size_options": self.image_size_options,
            "extra_view_configs": self.extra_view_configs,
            "pipeline": self.pipeline,
            "models": [model.to_dict() for model in self.models],
        }


def load_settings(path: Path = SETTINGS_PATH) -> UserSettings:
    if not path.is_file():
        return UserSettings()
    try:
        return UserSettings.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return UserSettings()


def save_settings(settings: UserSettings, path: Path = SETTINGS_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(settings.to_dict(), indent=2) + "\n", encoding="utf-8")
    return path


def configured_prediction_csv() -> Path | None:
    value = load_settings().prediction_csv.strip()
    return configured_path(value) if value else None


def configured_artifact_dirs() -> list[Path]:
    dirs: list[Path] = []
    for model in load_settings().models:
        if model.enabled and model.artifact_dir.strip():
            dirs.append(configured_path(model.artifact_dir))
    return dirs


def configured_model_sources() -> list[tuple[str, Path, Path]]:
    sources: list[tuple[str, Path, Path]] = []
    for model in load_settings().models:
        if not model.enabled:
            continue
        artifact_dir = configured_path(model.artifact_dir) if model.artifact_dir.strip() else Path()
        prediction_csv = configured_path(model.prediction_csv) if model.prediction_csv.strip() else Path()
        if artifact_dir or prediction_csv:
            sources.append((model.key, artifact_dir, prediction_csv))
    return sources


def configured_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        parts = path.parts
        if "AutoTorch" in parts:
            repo_index = parts.index("AutoTorch")
            source_repo = Path(*parts[: repo_index + 1])
            mapped = PACKAGE_ROOT.parent.joinpath(*parts[repo_index + 1 :])
            if source_repo != PACKAGE_ROOT.parent and (mapped.exists() or str(PACKAGE_ROOT.parent) == "/app"):
                return mapped
        if path.exists():
            return path
        return path
    return PACKAGE_ROOT.parent / path


def require_setting(value, name: str):
    if value in ("", None, [], ()):
        raise RuntimeError(f"Missing required setting: {name} in {SETTINGS_PATH}")
    return value
