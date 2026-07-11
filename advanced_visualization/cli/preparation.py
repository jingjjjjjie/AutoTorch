"""Prepare a model artifact directory for visualization.

For now the prediction CSV and known model runs are hardcoded through
MODEL_RUNS in config.py. This script creates a standard manifest plus a
prepared CSV that the visualization app can load without knowing run-specific
details.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from advanced_visualization.core.artifacts import build_manifest, save_manifest
from advanced_visualization.core.columns import infer_standard_columns
from advanced_visualization.core.config import PREPARED_CSV_NAME, ModelRunConfig, all_model_runs
from advanced_visualization.core.settings import configured_prediction_csv


def resolve_model_config(artifact_dir: Path, epoch: int, model_key: str = "") -> ModelRunConfig:
    model_runs = all_model_runs()
    if model_key:
        if model_key not in model_runs:
            raise ValueError(f"Unknown model key: {model_key}")
        return model_runs[model_key]

    checkpoint = artifact_dir.expanduser() / "checkpoints" / f"epoch_{epoch}.pt"
    matches = [config for config in model_runs.values() if config.checkpoint.expanduser() == checkpoint]
    if matches:
        return matches[0]

    known = "\n".join(f"  - {key}" for key in model_runs)
    raise ValueError(
        f"No hardcoded model config matches {checkpoint}.\n"
        f"Pass --model-key explicitly. Known keys:\n{known}"
    )


def prepare_dataframe(df: pd.DataFrame, standard_columns: dict[str, str]) -> pd.DataFrame:
    prepared = df.copy()
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
    df = pd.read_csv(pred_csv, low_memory=False)
    standard_columns = infer_standard_columns(df, model_config.prediction_column)
    if model_config.image_column:
        if model_config.image_column not in df.columns:
            raise ValueError(f"{model_config.key}: configured image_column {model_config.image_column!r} is missing from {pred_csv}")
        standard_columns["image_column"] = model_config.image_column
    prepared = prepare_dataframe(df, standard_columns)

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a visualization artifact directory.")
    parser.add_argument("--artifact-dir", required=True, type=Path, help="Run artifact directory, usually the folder containing checkpoints/.")
    parser.add_argument("--pred-csv", type=Path, default=configured_prediction_csv(), help="Prediction CSV to standardize.")
    parser.add_argument("--weights-epoch", type=int, required=True, help="Checkpoint epoch number, e.g. 11 for checkpoints/epoch_11.pt.")
    parser.add_argument("--model-key", default="", help="Explicit hardcoded model key from advanced_visualization.core.config.MODEL_RUNS.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.pred_csv is None:
        raise SystemExit("No prediction CSV configured. Set it in Settings or pass --pred-csv.")
    prepared_csv = prepare_artifact(
        artifact_dir=args.artifact_dir,
        pred_csv=args.pred_csv,
        epoch=args.weights_epoch,
        model_key=args.model_key,
    )
    print(f"Prepared visualization CSV: {prepared_csv}")
    print(f"Manifest: {args.artifact_dir / 'visualization_manifest.json'}")


if __name__ == "__main__":
    main()
