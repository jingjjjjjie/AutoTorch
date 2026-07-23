"""Router template for an exported Ench21 VAN Small data directory."""

from __future__ import annotations

import json
from pathlib import Path


ROUTER_API_VERSION = 1


def resolve(*, model_id: str, data_dir: Path) -> dict:
    manifest = json.loads(
        (data_dir / "visualization_manifest.json").read_text(encoding="utf-8")
    )
    prediction_data = Path(
        manifest.get("prepared_predictions_csv")
        or data_dir / "prepared_predictions.csv"
    )
    return {
        "framework": "tensorflow",
        "engine": "vansmall",
        "branch": manifest["branch"],
        "checkpoint": manifest["checkpoint"],
        "prediction_data": prediction_data,
        "feature_data": prediction_data,
        "model_name": "vansmall",
        "head_type": "tensorflow_export",
        "image_size": 512,
        "columns": {
            "sample_id": "uuid",
            "image": manifest["image_column"],
            "truth": "label",
            "prediction": manifest["prediction_column"],
            "subclass": "Recapture_Subclass",
        },
        "layers": [
            {
                "key": "norm3",
                "label": "norm3",
                "module": "norm3",
                "max_long_edge": 768,
            },
            {
                "key": "block3_3",
                "label": "block3.3",
                "module": "block3.3",
                "max_long_edge": 768,
            },
            {
                "key": "block4_1",
                "label": "block4.1",
                "module": "block4.1",
                "max_long_edge": 768,
            },
            {
                "key": "norm4",
                "label": "norm4 (final)",
                "module": "norm4",
                "final": True,
                "max_long_edge": 1024,
            },
        ],
    }
