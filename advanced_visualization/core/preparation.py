"""Preparation of standardized visualization artifact directories."""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from advanced_visualization.core.artifacts import build_manifest, save_manifest
from advanced_visualization.core.columns import infer_standard_columns
from advanced_visualization.core.config import (
    PREPARED_CSV_NAME,
    ModelRunConfig,
    all_model_runs,
)


def resolve_model_config(
    artifact_dir: Path, epoch: int, model_key: str = ""
) -> ModelRunConfig:
    model_runs = all_model_runs()
    if model_key:
        if model_key not in model_runs:
            raise ValueError(f"Unknown model key: {model_key}")
        return model_runs[model_key]

    checkpoint = artifact_dir.expanduser() / "checkpoints" / f"epoch_{epoch}.pt"
    matches = [
        config
        for config in model_runs.values()
        if config.checkpoint.expanduser() == checkpoint
    ]
    if matches:
        return matches[0]

    known = "\n".join(f"  - {key}" for key in model_runs)
    raise ValueError(
        f"No model config matches {checkpoint}.\n"
        f"Pass --model-key explicitly. Known keys:\n{known}"
    )


def prepare_dataframe(
    frame: pd.DataFrame, standard_columns: dict[str, str]
) -> pd.DataFrame:
    prepared = frame.copy()
    for standard_name, source_column in standard_columns.items():
        if source_column and source_column in prepared.columns:
            prepared[f"__{standard_name}"] = prepared[source_column]
    return prepared


def prepare_artifact(
    *,
    artifact_dir: Path,
    pred_csv: Path,
    epoch: int,
    model_key: str = "",
    copy_csv: bool = True,
) -> Path:
    artifact_dir = artifact_dir.expanduser()
    pred_csv = pred_csv.expanduser()
    if not pred_csv.is_file():
        raise FileNotFoundError(f"Prediction CSV does not exist: {pred_csv}")

    model_config = resolve_model_config(artifact_dir, epoch, model_key=model_key)
    frame = pd.read_csv(pred_csv, low_memory=False)
    standard_columns = infer_standard_columns(frame, model_config.prediction_column)
    if model_config.image_column:
        if model_config.image_column not in frame.columns:
            raise ValueError(
                f"{model_config.key}: configured image column "
                f"{model_config.image_column!r} is missing from {pred_csv}"
            )
        standard_columns["image_column"] = model_config.image_column
    prepared = prepare_dataframe(frame, standard_columns)

    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "gradcam").mkdir(parents=True, exist_ok=True)
    prepared_csv = artifact_dir / PREPARED_CSV_NAME
    if copy_csv:
        prepared.to_csv(prepared_csv, index=False)
    else:
        shutil.copy2(pred_csv, prepared_csv)

    manifest = build_manifest(
        artifact_dir=artifact_dir,
        source_csv=pred_csv,
        model_config=model_config,
        image_column=standard_columns["image_column"],
        item_id_column=standard_columns["item_id_column"],
        truth_column=standard_columns["truth_column"],
        prediction_column=standard_columns["prediction_column"],
        subclass_column=standard_columns["subclass_column"],
    )
    save_manifest(manifest)
    return prepared_csv
