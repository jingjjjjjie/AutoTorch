"""Tests for the FastAPI visualization platform."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

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


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("all", [0, 1, 2, 3]),
        ("failures", [0, 2]),
        ("false_positives", [0]),
        ("false_negatives", [2]),
        ("correct", [1]),
        ("high_confidence", [0]),
        ("low_confidence", []),
    ],
)
def test_all_failure_filter_modes(mode: str, expected: list[int]) -> None:
    frame = pd.DataFrame(
        {"__row_id": range(4), "label": [0, 1, 1, -1], "score": [0.95, 0.8, 0.2, 0.7]}
    )
    request = FilterRequest(
        source_id="test", truth_column="label", prediction_column="score",
        failure_mode=mode, high_confidence=0.9, low_confidence=0.6,
    )

    assert filter_frame(frame, request)["__row_id"].tolist() == expected


def test_search_category_filter_and_page_clamping() -> None:
    frame = pd.DataFrame(
        {
            "__row_id": range(4), "id": ["alpha", "beta", "alphabet", "gamma"],
            "group": ["a", "b", "a", "b"],
        }
    )
    request = FilterRequest(
        source_id="test", search="alpha", search_columns=["id"],
        categorical_filters={"group": ["a"]}, page=99, page_size=12,
    )
    filtered = filter_frame(frame, request)
    page, metadata = page_frame(filtered, request)

    assert page["__row_id"].tolist() == [0, 2]
    assert metadata["page"] == 1
    assert metadata["pages"] == 1


@pytest.mark.parametrize(
    ("truth_rows", "expected"),
    [
        ("all", [0, 1, 2, 3]),
        ("valid", [0, 1, 2]),
        ("positive", [1, 2]),
        ("negative", [0]),
        ("invalid", [3]),
    ],
)
def test_all_truth_row_modes(truth_rows: str, expected: list[int]) -> None:
    frame = pd.DataFrame(
        {"__row_id": range(4), "label": [0, 1, 1, -1], "score": [0.2, 0.8, 0.4, 0.7]}
    )
    request = FilterRequest(
        source_id="test", truth_column="label", prediction_column="score", truth_rows=truth_rows,
    )

    assert sorted(filter_frame(frame, request)["__row_id"].tolist()) == expected


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
    review = repository.review_dataframe(source.id)
    assert first is second
    assert first["model_feature_000"].dtype == np.float32
    assert "model_feature_000" not in review.columns

    pd.DataFrame({"id": [1, 2, 3], "model_feature_000": [0.1, 0.2, 0.3]}).to_csv(csv_path, index=False)
    reloaded = repository.dataframe(source.id)
    assert len(reloaded) == 3
    assert reloaded is not first


def test_schema_prefers_available_image_and_detects_montage(tmp_path: Path) -> None:
    from PIL import Image

    image_path = tmp_path / "crop.png"
    montage_path = tmp_path / "montage.png"
    Image.new("RGB", (20, 20)).save(image_path)
    Image.new("RGB", (20, 20)).save(montage_path)
    csv_path = tmp_path / "prepared.csv"
    pd.DataFrame(
        {
            "ori_path": ["missing.jpg"],
            "absolute_crop_path": [str(image_path)],
            "tf_crop_layer_montage_path": [str(montage_path)],
        }
    ).to_csv(csv_path, index=False)
    repository = FixedRepository(DataSource("source", "Source", csv_path, "unknown", None))

    schema = repository.schema("source")

    assert schema["defaults"]["image_column"] == "absolute_crop_path"
    assert schema["gradcam_columns"] == ["tf_crop_layer_montage_path"]
    assert schema["image_availability"]["ori_path"] == 0.0
    assert repository.schema("source") is schema


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

    service = ProjectionService()
    result = service.project(frame, request, "version")

    assert result["rows"] == 5
    assert {point["label"] for point in result["points"]} == {"a", "b"}
    assert all(np.isfinite(point["x"]) and np.isfinite(point["y"]) for point in result["points"])
    assert service.project(frame, request, "version") is result


def test_projection_rejects_invalid_inputs() -> None:
    frame = pd.DataFrame({"__row_id": range(3), "feature_0": [1.0, 2.0, 3.0]})
    service = ProjectionService()

    with pytest.raises(ValueError, match="at least two"):
        service.project(
            frame,
            ProjectionRequest(source_id="source", feature_columns=["feature_0"], max_rows=3),
            "version",
        )
    with pytest.raises(ValueError, match="No valid"):
        service.project(
            frame,
            ProjectionRequest(source_id="source", feature_columns=["missing"], max_rows=3),
            "version",
        )


def test_tsne_projection() -> None:
    frame = pd.DataFrame(
        {
            "__row_id": range(5),
            "feature_0": [0.0, 1.0, 2.0, 3.0, 4.0],
            "feature_1": [1.0, 0.0, 1.0, 0.0, 1.0],
        }
    )
    result = ProjectionService().project(
        frame,
        ProjectionRequest(
            source_id="source", method="tsne", feature_columns=["feature_0", "feature_1"],
            perplexity=2, max_rows=5,
        ),
        "version",
    )

    assert result["rows"] == 5
    assert all(np.isfinite(point["x"]) for point in result["points"])


def test_image_service_creates_bounded_jpeg(tmp_path: Path) -> None:
    from PIL import Image

    path = tmp_path / "source.png"
    Image.new("RGB", (800, 400), color=(40, 120, 80)).save(path)

    content, etag = image_bytes(path, max_side=200)

    assert content.startswith(b"\xff\xd8")
    assert etag.isdigit()
    decoded = Image.open(__import__("io").BytesIO(content))
    assert decoded.size == (200, 100)


def test_image_service_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        image_bytes(tmp_path / "missing.png")


def test_http_routes_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from PIL import Image
    from advanced_visualization.web import app as web_app

    image_path = tmp_path / "image.png"
    Image.new("RGB", (30, 20), color=(20, 80, 120)).save(image_path)
    csv_path = tmp_path / "source.csv"
    pd.DataFrame(
        {
            "id": ["a", "b", "c"], "label": [0, 1, 1],
            "score": [0.9, 0.8, 0.2], "image_path": [str(image_path)] * 3,
            "feature_0": [0.0, 1.0, 2.0], "feature_1": [2.0, 1.0, 0.0],
        }
    ).to_csv(csv_path, index=False)
    fixed = FixedRepository(DataSource("source", "Source", csv_path, "model", None))
    monkeypatch.setattr(web_app, "repository", fixed)

    assert web_app.health() == {"status": "ok"}
    assert web_app.favicon().status_code == 204
    assert web_app.sources()[0].id == "source"
    schema_response = web_app.schema("source")
    assert schema_response.feature_columns == ["feature_0", "feature_1"]

    review_response = web_app.review(
        FilterRequest(
            source_id="source", item_id_column="id", image_column="image_path",
            truth_column="label", prediction_column="score", page_size=12,
        )
    )
    first_row = review_response["rows"][0]
    assert first_row["image_url"].endswith("max_side=480")
    image_response = web_app.image("source", first_row["row_id"], "image_path", 480)
    assert image_response.media_type == "image/jpeg"

    with pytest.raises(web_app.HTTPException) as missing:
        web_app.image("source", 99, "image_path", 480)
    assert missing.value.status_code == 404

    projection_response = web_app.projection(
        ProjectionRequest(
            source_id="source", feature_columns=["feature_0", "feature_1"],
            image_column="image_path", max_rows=3,
        )
    )
    assert projection_response["rows"] == 3
