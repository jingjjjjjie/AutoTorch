"""Generate one router-registered PyTorch Grad-CAM++ layer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from advanced_visualization.core.registered_gradcam import generate_pytorch_layer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_id")
    parser.add_argument("--layer", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = generate_pytorch_layer(
        args.model_id,
        args.layer,
        limit=args.limit,
        overwrite=args.overwrite,
        artifact_root=args.artifact_root,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
