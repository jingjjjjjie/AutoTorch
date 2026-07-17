"""Tests for the FastAPI visualization platform."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from advanced_visualization.web.filtering import filter_frame, page_frame
from advanced_visualization.web.images import image_bytes
from advanced_visualization.web.models import FilterRequest, ProjectionRequest
from advanced_visualization.web.projections import ProjectionService
from advanced_visualization.web.repository import DataSource, DatasetRepository


def test_filtering_classifies_failures_and_pages() -> None:
    frame = pd.DataFrame(
        {
            "__row_id": [0, 1, 2, 3],
            "id": ["a", "b", "c", "d"],
            "label": [0, 1, 1, -1],
            "score": [0.9, 0.8, 0.2, 0.7],
            "group": ["one", "one", "two", "two"],
        }
    )
    request = FilterRequest(
        source_id="test",
        truth_column="label",
        prediction_column="score",
        failure_mode="failures",
        page_size=12,
    )

    filtered = filter_frame(frame, request)
    page, metadata = page_frame(filtered, request)

    assert page["__row_id"].tolist() == [0, 2]
    assert page["__failure_type"].tolist() == ["false positive", "false negative"]
    assert metadata["metrics"]["failure_rate"] == 1.0


class FixedRepository(DatasetRepository):
    def __init__(self, source: DataSource) -> None:
        super().__init__()
        self._source = source

    def sources(self) -> list[DataSource]:
        return [self._source]


def test_repository_caches_and_reloads_float32_features(tmp_path: Path) -> None:
    csv_path = tmp_path / "features.csv"
    pd.DataFrame({"id": [1, 2], "model_feature_000": [0.1, 0.2]}).to_csv(csv_path, index=False)
    source = DataSource("source", "Source", csv_path, "model", None)
    repository = FixedRepository(source)

    first = repository.dataframe(source.id)
    second = repository.dataframe(source.id)
    assert first is second
    assert first["model_feature_000"].dtype == np.float32

    pd.DataFrame({"id": [1, 2, 3], "model_feature_000": [0.1, 0.2, 0.3]}).to_csv(csv_path, index=False)
    reloaded = repository.dataframe(source.id)
    assert len(reloaded) == 3
    assert reloaded is not first


def test_projection_returns_finite_points() -> None:
    frame = pd.DataFrame(
        {
            "__row_id": range(5),
            "feature_0": [0.0, 1.0, 2.0, 3.0, 4.0],
            "feature_1": [2.0, 1.0, 0.0, 1.0, 2.0],
            "group": ["a", "a", "b", "b", "b"],
        }
    )
    request = ProjectionRequest(
        source_id="source",
        feature_columns=["feature_0", "feature_1"],
        color_column="group",
        max_rows=5,
    )

    result = ProjectionService().project(frame, request, "version")

    assert result["rows"] == 5
    assert {point["label"] for point in result["points"]} == {"a", "b"}
    assert all(np.isfinite(point["x"]) and np.isfinite(point["y"]) for point in result["points"])


def test_image_service_creates_bounded_jpeg(tmp_path: Path) -> None:
    from PIL import Image

    path = tmp_path / "source.png"
    Image.new("RGB", (800, 400), color=(40, 120, 80)).save(path)

    content, etag = image_bytes(path, max_side=200)

    assert content.startswith(b"\xff\xd8")
    assert etag.isdigit()
    decoded = Image.open(__import__("io").BytesIO(content))
    assert decoded.size == (200, 100)
