"""Router template for an AutoTorch UniRepLKNet experiment directory."""

from __future__ import annotations

import json
from pathlib import Path

import yaml


ROUTER_API_VERSION = 1


def resolve(*, model_id: str, data_dir: Path) -> dict:
    manifest = json.loads(
        (data_dir / "visualization_manifest.json").read_text(encoding="utf-8")
    )
    config = yaml.safe_load((data_dir / "config.yaml").read_text(encoding="utf-8"))
    overrides_path = data_dir / "router_overrides.json"
    overrides = (
        json.loads(overrides_path.read_text(encoding="utf-8"))
        if overrides_path.is_file()
        else {}
    )
    prediction_data = Path(
        overrides.get("prediction_data")
        or manifest.get("prepared_csv")
        or data_dir / "prepared_predictions.csv"
    )
    columns = {
        "sample_id": manifest.get("item_id_column", "uuid"),
        "image": manifest["image_column"],
        "truth": manifest.get("truth_column", "label"),
        "prediction": overrides.get("prediction_column")
        or manifest["prediction_column"],
        "subclass": manifest.get("subclass_column", "Recapture_Subclass"),
    }
    return {
        "framework": "pytorch",
        "engine": "unireplknet",
        "checkpoint": overrides.get("checkpoint") or manifest["checkpoint"],
        "prediction_data": prediction_data,
        "feature_data": overrides.get("feature_data") or prediction_data,
        "model_name": config["model"]["backbone_name"],
        "head_type": config["model"]["head_type"],
        "image_size": int(config["transform"]["image_size"]),
        "columns": columns,
        "layers": [
            {
                "key": "stage1",
                "label": "Stage 1",
                "module": "feature_extractor.stages.0",
                "max_long_edge": 768,
            },
            {
                "key": "stage2",
                "label": "Stage 2",
                "module": "feature_extractor.stages.1",
                "max_long_edge": 768,
            },
            {
                "key": "stage3",
                "label": "Stage 3",
                "module": "feature_extractor.stages.2",
                "max_long_edge": 768,
            },
            {
                "key": "stage4",
                "label": "Stage 4 (final)",
                "module": "feature_extractor.stages.3",
                "final": True,
                "max_long_edge": 1024,
            },
        ],
        "review_preset": overrides.get("review_preset", {}),
    }
