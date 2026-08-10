"""Verify the June feature and final-layer Grad-CAM refresh."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from advanced_visualization.core.feature_data import FEATURE_PATTERN
from advanced_visualization.core.gradcam_cache import gradcam_cache_candidates
from advanced_visualization.core.images import valid_image
from advanced_visualization.core.model_router import registered_model_routes
from advanced_visualization.core.registered_gradcam import TARGETS


EXPECTED_ROWS = 51_403
MODEL_IDS = (
    "Ex8point2_UniRepLKNet_T_legacy_v1_512_ori_epoch10",
    "Ex8point4_UniRepLKNet_B_in22k_legacy_v1_512_crop_epoch7",
    "Ex8point2res1024_moredata_largerbs_UniRepLKNet_T_legacy_v1_1024_ori_epoch11",
    "square_exp2_Ex8point2res1024_largerbs_21plusaugfeb_UniRepLKNet_T_legacy_v1_1024_ori_epoch8",
    "ench21_vansmall_ori",
    "ench21_vansmall_crop",
    "InternEnch_Ex8point2res1024largerb",
)
STATE_PATH = Path("/mnt4/advanced_visualization/june_refresh_verification.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("features", "all"), default="all")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    routes = registered_model_routes()
    results: dict[str, dict] = {}
    errors: list[str] = []
    for model_id in MODEL_IDS:
        route = routes.get(model_id)
        if route is None:
            errors.append(f"{model_id}: route is not registered")
            continue
        if route.feature_data is None or not route.feature_data.is_file():
            errors.append(f"{model_id}: feature CSV is missing")
            continue
        frame = pd.read_csv(route.feature_data, low_memory=False)
        feature_columns = [
            column for column in frame.columns if FEATURE_PATTERN.search(str(column))
        ]
        prediction_column = route.columns["prediction"]
        image_column = route.columns["image"]
        result = {
            "rows": len(frame),
            "feature_dimensions": len(feature_columns),
            "image_column": image_column,
            "image_column_present": image_column in frame,
            "missing_feature_values": int(frame[feature_columns].isna().sum().sum())
            if feature_columns
            else None,
            "missing_predictions": int(frame[prediction_column].isna().sum())
            if prediction_column in frame
            else None,
        }
        if len(frame) != EXPECTED_ROWS:
            errors.append(f"{model_id}: expected {EXPECTED_ROWS} rows, found {len(frame)}")
        if not feature_columns:
            errors.append(f"{model_id}: no feature columns")
        elif result["missing_feature_values"]:
            errors.append(f"{model_id}: feature values are missing")
        if prediction_column not in frame or result["missing_predictions"]:
            errors.append(f"{model_id}: predictions are missing")
        if image_column not in frame:
            errors.append(f"{model_id}: image column {image_column!r} is missing")

        if args.phase == "all" and image_column in frame:
            final_layers = route.final_layers
            if not final_layers:
                errors.append(f"{model_id}: no final Grad-CAM layer")
            images = []
            seen = set()
            for value in frame[image_column]:
                image = valid_image(value)
                if image is None:
                    continue
                key = str(image.resolve())
                if key not in seen:
                    seen.add(key)
                    images.append(image)
            missing = 0
            for layer in final_layers:
                for image in images:
                    for target in TARGETS:
                        candidates = gradcam_cache_candidates(
                            route.artifact_dir / "gradcam",
                            image,
                            method="gradcam++",
                            target=target,
                            layer=layer.key,
                        )
                        if not any(candidate.is_file() for candidate in candidates):
                            missing += 1
            result["unique_valid_images"] = len(images)
            result["missing_final_gradcam_files"] = missing
            if missing:
                errors.append(f"{model_id}: {missing} final Grad-CAM files are missing")
        results[model_id] = result

    payload = {
        "status": "complete" if not errors else "failed",
        "phase": args.phase,
        "expected_rows": EXPECTED_ROWS,
        "models": results,
        "errors": errors,
    }
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATE_PATH.with_name(f".{STATE_PATH.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, STATE_PATH)
    print(json.dumps(payload, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
