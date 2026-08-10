"""Atomically add a CSV column as an exact alias of an existing column."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    return parser.parse_args()


def ensure_alias(csv_path: Path, source: str, target: str) -> bool:
    temporary = csv_path.with_name(f".{csv_path.name}.{os.getpid()}.tmp")
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as source_handle:
            reader = csv.reader(source_handle)
            header = next(reader)
            if target in header:
                return False
            if source not in header:
                raise ValueError(f"Source column {source!r} is missing from {csv_path}")
            source_index = header.index(source)
            with temporary.open("w", encoding="utf-8", newline="") as output_handle:
                writer = csv.writer(output_handle)
                writer.writerow([*header, target])
                for row in reader:
                    if len(row) != len(header):
                        raise ValueError(
                            f"Malformed row with {len(row)} fields; expected {len(header)}"
                        )
                    writer.writerow([*row, row[source_index]])
        os.replace(temporary, csv_path)
        return True
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    changed = ensure_alias(args.csv_path, args.source, args.target)
    action = "Added" if changed else "Already present"
    print(f"{action}: {args.target} in {args.csv_path}")


if __name__ == "__main__":
    main()
