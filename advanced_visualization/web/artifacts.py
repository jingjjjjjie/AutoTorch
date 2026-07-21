"""Web-facing URLs and prepared CAM artifact resolution."""
from __future__ import annotations

from pathlib import Path
from urllib.parse import quote

import pandas as pd

from advanced_visualization.core.gradcam_cache import gradcam_cache_candidates
from advanced_visualization.core.images import valid_image
from advanced_visualization.web.repository import DataSource


def image_url(source_id: str, row_id: int | None, column: str, max_side: int = 480) -> str:
    if row_id is None or not column:
        return ""
    return f"/api/images/{source_id}/{row_id}?column={quote(column, safe='')}&max_side={max_side}"


def prepared_gradcam_path(
    source: DataSource,
    row: pd.Series,
    image_column: str,
    method: str,
    target: str = "fraud",
) -> Path | None:
    if not source.artifact_dir or image_column not in row.index:
        return None
    image_path = valid_image(row[image_column])
    if image_path is None:
        return None
    for candidate in gradcam_cache_candidates(
        source.artifact_dir / "gradcam", image_path, method=method, target=target
    ):
        if candidate.is_file():
            return candidate
    return None


def prepared_gradcam_url(
    source_id: str,
    row_id: int,
    image_column: str,
    method: str,
    target: str = "fraud",
) -> str:
    return (
        f"/api/gradcam/{source_id}/{row_id}?image_column={quote(image_column, safe='')}"
        f"&method={quote(method, safe='')}&target={quote(target, safe='')}"
    )


def gradcam_url(
    source: DataSource,
    row: pd.Series,
    row_id: int | None,
    *,
    image_column: str,
    gradcam_column: str = "",
    method: str = "gradcam",
    target: str = "fraud",
    max_side: int = 480,
) -> str:
    if row_id is None:
        return ""
    if gradcam_column and target == "fraud" and gradcam_column in row.index and valid_image(row[gradcam_column]):
        return image_url(source.id, row_id, gradcam_column, max_side=max_side)
    if image_column and prepared_gradcam_path(source, row, image_column, method, target):
        return prepared_gradcam_url(source.id, row_id, image_column, method, target)
    return ""

