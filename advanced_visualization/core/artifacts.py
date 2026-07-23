"""Artifact manifest helpers for the visualization app."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from advanced_visualization.core.config import (
    MANIFEST_NAME,
    PREPARED_CSV_NAME,
    ModelRunConfig,
)
from advanced_visualization.core.settings import configured_model_sources, configured_prediction_csv


@dataclass(frozen=True)
class VisualizationManifest:
    artifact_dir: Path
    prepared_csv: Path
    source_csv: Path
    model_key: str
    checkpoint: Path
    gradcam_dir: Path
    image_column: str = ""
    item_id_column: str = ""
    truth_column: str = ""
    prediction_column: str = ""
    subclass_column: str = ""

    def to_json_dict(self) -> dict:
        return {
            "artifact_dir": str(self.artifact_dir),
            "prepared_csv": str(self.prepared_csv),
            "source_csv": str(self.source_csv),
            "model_key": self.model_key,
            "checkpoint": str(self.checkpoint),
            "gradcam_dir": str(self.gradcam_dir),
            "image_column": self.image_column,
            "item_id_column": self.item_id_column,
            "truth_column": self.truth_column,
            "prediction_column": self.prediction_column,
            "subclass_column": self.subclass_column,
        }

    @classmethod
    def from_json_dict(cls, payload: dict) -> "VisualizationManifest":
        artifact_dir = Path(payload.get("artifact_dir") or payload.get("artifact_root") or "")
        prepared_csv = Path(payload.get("prepared_csv") or payload.get("prepared_predictions_csv") or payload.get("prepared_predictions") or artifact_dir / PREPARED_CSV_NAME)
        model_key = str(payload.get("model_key") or payload.get("model_id") or payload.get("run_name") or artifact_dir.name)
        gradcam_dir = Path(payload.get("gradcam_dir") or artifact_dir / "gradcam")
        columns = dict(payload.get("columns") or {})
        return cls(
            artifact_dir=artifact_dir,
            prepared_csv=prepared_csv,
            source_csv=Path(payload.get("source_csv") or ""),
            model_key=model_key,
            checkpoint=Path(payload.get("checkpoint") or ""),
            gradcam_dir=gradcam_dir,
            image_column=str(payload.get("image_column") or columns.get("image") or ""),
            item_id_column=str(payload.get("item_id_column") or columns.get("sample_id") or ""),
            truth_column=str(payload.get("truth_column") or columns.get("truth") or ""),
            prediction_column=str(payload.get("prediction_column") or columns.get("prediction") or ""),
            subclass_column=str(payload.get("subclass_column") or columns.get("subclass") or ""),
        )


def manifest_path(artifact_dir: Path) -> Path:
    return artifact_dir.expanduser() / MANIFEST_NAME


def prepared_csv_path(artifact_dir: Path) -> Path:
    return artifact_dir.expanduser() / PREPARED_CSV_NAME


def load_manifest(artifact_dir: Path) -> Optional[VisualizationManifest]:
    path = manifest_path(artifact_dir)
    if not path.is_file():
        return None
    return VisualizationManifest.from_json_dict(json.loads(path.read_text(encoding="utf-8")))


def save_manifest(manifest: VisualizationManifest) -> Path:
    path = manifest_path(manifest.artifact_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest.to_json_dict(), indent=2) + "\n", encoding="utf-8")
    return path


def build_manifest(
    *,
    artifact_dir: Path,
    source_csv: Path,
    model_config: ModelRunConfig,
    image_column: str = "",
    item_id_column: str = "",
    truth_column: str = "",
    prediction_column: str = "",
    subclass_column: str = "",
) -> VisualizationManifest:
    artifact_dir = artifact_dir.expanduser()
    return VisualizationManifest(
        artifact_dir=artifact_dir,
        prepared_csv=prepared_csv_path(artifact_dir),
        source_csv=source_csv.expanduser(),
        model_key=model_config.key,
        checkpoint=model_config.checkpoint,
        gradcam_dir=artifact_dir / "gradcam",
        image_column=image_column,
        item_id_column=item_id_column,
        truth_column=truth_column,
        prediction_column=prediction_column or model_config.prediction_column,
        subclass_column=subclass_column,
    )


def default_csv_paths() -> list[Path]:
    configured_paths = []
    for _model_key, artifact_dir, prediction_csv in configured_model_sources():
        if artifact_dir is not None:
            manifest = load_manifest(artifact_dir)
            if manifest and manifest.prepared_csv.exists():
                configured_paths.append(manifest.prepared_csv)
                continue
            prepared_csv = prepared_csv_path(artifact_dir)
            if prepared_csv.exists():
                configured_paths.append(prepared_csv)
                continue
        if prediction_csv is not None and prediction_csv.is_file():
            configured_paths.append(prediction_csv)
    if configured_paths:
        return list(dict.fromkeys(configured_paths))

    configured_csv = configured_prediction_csv()
    return [configured_csv] if configured_csv else []


def available_data_sources() -> list[dict[str, object]]:
    sources: list[dict[str, object]] = []
    seen_prepared_paths: set[Path] = set()
    added_model_source = False
    configured_csv = configured_prediction_csv()

    def append_prepared(label: str, path: Path, artifact_dir: Path | None, model_key: str) -> None:
        resolved = path.expanduser()
        if resolved in seen_prepared_paths:
            return
        seen_prepared_paths.add(resolved)
        sources.append({"label": label, "path": resolved, "artifact_dir": artifact_dir, "model_key": model_key})

    for model_key, artifact_dir, prediction_csv in configured_model_sources():
        label = model_key
        manifest = load_manifest(artifact_dir) if artifact_dir is not None else None
        if manifest and manifest.prepared_csv.exists():
            append_prepared(f"{label} - prepared", manifest.prepared_csv, artifact_dir, model_key)
            continue

        prepared_csv = prepared_csv_path(artifact_dir) if artifact_dir is not None else None
        if prepared_csv is not None and prepared_csv.is_file():
            append_prepared(f"{label} - prepared", prepared_csv, artifact_dir, model_key)
            continue

        if prediction_csv is not None and prediction_csv.is_file():
            added_model_source = True
            sources.append(
                {
                    "label": f"{label} - source",
                    "path": prediction_csv.expanduser(),
                    "artifact_dir": artifact_dir,
                    "model_key": model_key,
                }
            )

    if configured_csv and configured_csv.exists() and not added_model_source:
        sources.append({"label": f"{configured_csv.name} - source", "path": configured_csv, "artifact_dir": None, "model_key": ""})
    return sources


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)
