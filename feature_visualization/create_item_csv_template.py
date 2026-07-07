"""Create an item-level CSV template for the UniRepLKNet-T feature explorer."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


METADATA_COLUMNS = [
    "id",
    "path",
    "class",
    "source",
    "batch",
    "sample_type",
    "split",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="Template CSV path to write.")
    parser.add_argument("--feature-dim", default=640, type=int, help="Number of UniRepLKNet-T feature columns.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    feature_columns = [f"feature_{index:04d}" for index in range(args.feature_dim)]
    example = pd.DataFrame(
        [
            {
                "id": "item_000001",
                "path": "/absolute/path/to/image.jpg",
                "class": "collected colour printed",
                "source": "collected",
                "batch": "batch_01",
                "sample_type": "colour printed",
                "split": "eval",
                **{column: "" for column in feature_columns},
            }
        ],
        columns=METADATA_COLUMNS + feature_columns,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    example.to_csv(args.output, index=False)
    print(f"Wrote item-level CSV template to {args.output}")


if __name__ == "__main__":
    main()
