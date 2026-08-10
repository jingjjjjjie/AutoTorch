"""Export predictions and embeddings for one registered PyTorch model."""

from __future__ import annotations

import argparse
from pathlib import Path

from advanced_visualization.core.config import model_run
from advanced_visualization.core.feature_extraction import (
    extract_features_and_predictions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--image-column", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--incremental-from", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = extract_features_and_predictions(
        config=model_run(args.model_id),
        csv_path=args.csv,
        image_column=args.image_column,
        output_csv=args.output,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        incremental_from=args.incremental_from,
    )
    print(f"Wrote registered features and predictions to {output}")


if __name__ == "__main__":
    main()
