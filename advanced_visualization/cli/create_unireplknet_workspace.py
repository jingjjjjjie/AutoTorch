"""Create a router-backed UniRepLKNet visualization workspace."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
ROUTER_TEMPLATE = (
    REPO_ROOT / "advanced_visualization" / "router_templates" / "artifact_manifest.py"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--prediction-column", required=True)
    parser.add_argument("--image-column", default="absolute_ori_path")
    parser.add_argument("--branch", choices=("ori", "crop"), default="ori")
    parser.add_argument("--truth-column", default="label")
    parser.add_argument("--subclass-column", default="Recapture_Subclass")
    parser.add_argument("--model-name", default="unireplknet_t")
    parser.add_argument("--head-type", default="legacy_v1")
    parser.add_argument("--image-size", type=int, default=1024)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    args.workspace.mkdir(parents=True, exist_ok=True)
    (args.workspace / "features").mkdir(exist_ok=True)
    (args.workspace / "gradcam").mkdir(exist_ok=True)
    prepared = args.workspace / "features" / "prepared_predictions.csv"
    layers = [
        {
            "key": f"stage{index + 1}",
            "label": f"Stage {index + 1}" + (" (final)" if index == 3 else ""),
            "module": f"feature_extractor.stages.{index}",
            "final": index == 3,
            "max_long_edge": 1024 if index == 3 else 768,
        }
        for index in range(4)
    ]
    manifest = {
        "schema_version": 2,
        "model_id": args.model_id,
        "router": str(args.workspace / "router.py"),
        "data_dir": str(args.workspace),
        "artifact_dir": str(args.workspace),
        "prepared_predictions": str(prepared),
        "features": str(prepared),
        "checkpoint": str(args.checkpoint),
        "framework": "pytorch",
        "engine": "unireplknet",
        "branch": args.branch,
        "model_name": args.model_name,
        "head_type": args.head_type,
        "image_size": args.image_size,
        "columns": {
            "sample_id": "uuid",
            "image": args.image_column,
            "truth": args.truth_column,
            "prediction": args.prediction_column,
            "subclass": args.subclass_column,
        },
        "gradcam": {
            "method": "gradcam++",
            "targets": ["genuine", "fraud"],
            "layers": layers,
            "format": "webp",
            "quality": 80,
            "preserve_aspect_ratio": True,
            "allow_upscale": False,
            "original_image": "separate_reference",
            "quantitative_saliency_data": False,
        },
        "review": {"prepared_gradcam_layers": ["stage4"]},
    }
    temporary = args.workspace / ".visualization_manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.workspace / "visualization_manifest.json")
    shutil.copyfile(ROUTER_TEMPLATE, args.workspace / "router.py")
    print(f"Created {args.model_id} at {args.workspace}")


if __name__ == "__main__":
    main()
