"""Pre-generate Grad-CAM overlays for visualization CSVs."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from advanced_visualization.core.gradcam_generation import (
    GradcamGenerationOptions,
    pregenerate_csv,
    resolve_csv_paths,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        action="append",
        default=[],
        help="CSV file or artifact directory. Defaults to configured Settings artifacts.",
    )
    parser.add_argument("--image-column", default=None)
    parser.add_argument(
        "--filter",
        action="append",
        default=[],
        help="Filter rows with COLUMN=VALUE. Can be repeated.",
    )
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--save-workers", type=int, default=8)
    parser.add_argument("--max-output-side", type=int, default=None)
    parser.add_argument(
        "--cam-method",
        action="append",
        choices=["gradcam", "gradcam++"],
        default=None,
    )
    parser.add_argument(
        "--cam-target",
        action="append",
        choices=["fraud", "genuine"],
        default=None,
    )
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--max-error-examples", type=int, default=10)
    parser.add_argument(
        "--only-missing", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    return parser.parse_args()


def generation_options(args: argparse.Namespace) -> GradcamGenerationOptions:
    return GradcamGenerationOptions(
        image_column=args.image_column,
        filters=tuple(args.filter),
        offset=args.offset,
        limit=args.limit,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        save_workers=args.save_workers,
        max_output_side=args.max_output_side,
        output_root=args.output_root,
        cam_methods=tuple(args.cam_method or ["gradcam"]),
        cam_targets=tuple(args.cam_target or ["fraud"]),
        max_error_examples=args.max_error_examples,
        only_missing=args.only_missing,
        dry_run=args.dry_run,
        stop_on_error=args.stop_on_error,
    )


def main() -> None:
    args = parse_args()
    options = generation_options(args)
    try:
        options.validate()
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    csv_paths = resolve_csv_paths(args.csv)
    if not csv_paths:
        raise SystemExit("No configured CSVs found.")

    totals = [pregenerate_csv(path, options) for path in csv_paths]
    generated = sum(item[0] for item in totals)
    skipped = sum(item[1] for item in totals)
    failed = sum(item[2] for item in totals)
    print(f"Total: generated={generated}, skipped={skipped}, failed={failed}")


if __name__ == "__main__":
    main()
