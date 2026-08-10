"""Router for a self-contained model directory under the artifact root."""

from __future__ import annotations

import json
from pathlib import Path


ROUTER_API_VERSION = 1


def resolve(*, model_id: str, data_dir: Path) -> dict:
    """Resolve every active path from one readable visualization manifest."""
    manifest = json.loads(
        (data_dir / "visualization_manifest.json").read_text(encoding="utf-8")
    )
    columns = dict(manifest.get("columns") or {})
    gradcam = dict(manifest.get("gradcam") or {})
    review = dict(manifest.get("review") or {})
    montage = dict(review.get("montage") or {})
    return {
        "framework": manifest["framework"],
        "engine": manifest["engine"],
        "branch": manifest.get("branch", ""),
        "checkpoint": manifest["checkpoint"],
        "prediction_data": manifest["prepared_predictions"],
        "feature_data": manifest.get("features") or manifest["prepared_predictions"],
        "model_name": manifest["model_name"],
        "head_type": manifest["head_type"],
        "image_size": int(manifest["image_size"]),
        "columns": {
            "sample_id": columns.get("sample_id", "uuid"),
            "image": columns["image"],
            "truth": columns.get("truth", "label"),
            "prediction": columns["prediction"],
            "subclass": columns.get("subclass", "Recapture_Subclass"),
        },
        "layers": list(gradcam.get("layers") or []),
        "prepared_gradcam_layers": review.get("prepared_gradcam_layers"),
        "gradcam_montage_column": montage.get("column", ""),
        "gradcam_montage_layers": list(montage.get("layers") or []),
        "review_preset": dict(manifest.get("review_preset") or {}),
    }
