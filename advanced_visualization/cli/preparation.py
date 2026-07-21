"""Prepare a standardized model artifact directory for visualization."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from advanced_visualization.core.preparation import prepare_artifact
from advanced_visualization.core.settings import configured_prediction_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--pred-csv", type=Path, default=configured_prediction_csv())
    parser.add_argument("--weights-epoch", type=int, required=True)
    parser.add_argument("--model-key", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.pred_csv is None:
        raise SystemExit(
            "No prediction CSV configured. Set it in Settings or pass --pred-csv."
        )
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
