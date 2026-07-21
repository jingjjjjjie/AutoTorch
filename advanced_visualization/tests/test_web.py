"""Tests for the FastAPI visualization platform."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from advanced_visualization.core.gradcam_cache import (
    gradcam_cache_candidates,
    gradcam_file_index,
)
from advanced_visualization.core.heatmap import jet_overlay
from advanced_visualization.core.images import image_cache_digest
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


def test_filter_request_accepts_genuine_cam_target() -> None:
    request = FilterRequest(source_id="test", gradcam_target="genuine")

    assert request.gradcam_target == "genuine"


def test_gradcam_cache_separates_fraud_and_genuine_targets(tmp_path: Path) -> None:
    image_path = tmp_path / "source.jpg"
    image_path.write_bytes(b"image")
    fraud = gradcam_cache_candidates(
        tmp_path, image_path, method="gradcam", target="fraud"
    )[0]
    genuine = gradcam_cache_candidates(
        tmp_path, image_path, method="gradcam", target="genuine"
    )[0]
    fraud.write_bytes(b"fraud")
    genuine.write_bytes(b"genuine")

    assert fraud.name.endswith("_gradcam_logit.png")
    assert genuine.name.endswith("_gradcam_genuine_logit.png")
    digest = fraud.name.split("_", 1)[0]
    assert gradcam_file_index(str(tmp_path), method="gradcam", target="fraud") == {
        digest: str(fraud)
    }
    assert gradcam_file_index(str(tmp_path), method="gradcam", target="genuine") == {
        digest: str(genuine)
    }


def test_jet_overlay_maps_low_to_blue_and_high_to_red() -> None:
    base = np.zeros((1, 2, 3), dtype=np.uint8)
    overlay = np.asarray(jet_overlay(base, np.array([[0.0, 1.0]], dtype=np.float32)))

    assert overlay[0, 0, 2] > overlay[0, 0, 0]
    assert overlay[0, 1, 0] > overlay[0, 1, 2]


def test_image_cache_digest_changes_when_image_changes(tmp_path: Path) -> None:
    image_path = tmp_path / "image.jpg"
    image_path.write_bytes(b"first")
    first = image_cache_digest(image_path)

    image_path.write_bytes(b"a different image payload")

    assert image_cache_digest(image_path) != first


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
        source_id="test",
        truth_column="label",
        prediction_column="score",
        failure_mode=mode,
        high_confidence=0.9,
        low_confidence=0.6,
    )

    assert filter_frame(frame, request)["__row_id"].tolist() == expected


def test_search_category_filter_and_page_clamping() -> None:
    frame = pd.DataFrame(
        {
            "__row_id": range(4),
            "id": ["alpha", "beta", "alphabet", "gamma"],
            "group": ["a", "b", "a", "b"],
        }
    )
    request = FilterRequest(
        source_id="test",
        search="alpha",
        search_columns=["id"],
        categorical_filters={"group": ["a"]},
        page=99,
        page_size=12,
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
        source_id="test",
        truth_column="label",
        prediction_column="score",
        truth_rows=truth_rows,
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
    pd.DataFrame({"id": [1, 2], "model_feature_000": [0.1, 0.2]}).to_csv(
        csv_path, index=False
    )
    source = DataSource("source", "Source", csv_path, "model", None)
    repository = FixedRepository(source)

    first = repository.dataframe(source.id)
    second = repository.dataframe(source.id)
    review = repository.review_dataframe(source.id)
    assert first is second
    assert first["model_feature_000"].dtype == np.float32
    assert "model_feature_000" not in review.columns

    pd.DataFrame({"id": [1, 2, 3], "model_feature_000": [0.1, 0.2, 0.3]}).to_csv(
        csv_path, index=False
    )
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
    repository = FixedRepository(
        DataSource("source", "Source", csv_path, "unknown", None)
    )

    schema = repository.schema("source")

    assert schema["defaults"]["image_column"] == "absolute_crop_path"
    assert schema["gradcam_columns"] == ["tf_crop_layer_montage_path"]
    assert schema["image_availability"]["ori_path"] == 0.0
    assert repository.schema("source") is schema


def test_schema_cache_invalidates_when_gradcam_artifacts_change(tmp_path: Path) -> None:
    csv_path = tmp_path / "prepared.csv"
    artifact_dir = tmp_path / "artifacts"
    gradcam_dir = artifact_dir / "gradcam"
    gradcam_dir.mkdir(parents=True)
    pd.DataFrame({"id": ["one"], "label": [0], "score": [0.1]}).to_csv(
        csv_path, index=False
    )
    repository = FixedRepository(
        DataSource("source", "Source", csv_path, "unknown", artifact_dir)
    )

    initial = repository.schema("source")
    (gradcam_dir / "one_gradcam_logit.png").write_bytes(b"prepared")
    refreshed = repository.schema("source")

    assert initial["prepared_gradcam_methods"] == []
    assert refreshed["prepared_gradcam_methods"] == ["gradcam"]
    assert refreshed is not initial


def test_projection_returns_finite_points() -> None:
    frame = pd.DataFrame(
        {
            "__row_id": range(5),
            "uuid": [f"uuid-{index}" for index in range(5)],
            "feature_0": [0.0, 1.0, 2.0, 3.0, 4.0],
            "feature_1": [2.0, 1.0, 0.0, 1.0, 2.0],
            "group": ["a", "a", "b", "b", "b"],
        }
    )
    request = ProjectionRequest(
        source_id="source",
        feature_columns=["feature_0", "feature_1"],
        item_id_column="uuid",
        color_column="group",
        max_rows=5,
    )

    service = ProjectionService()
    result = service.project(frame, request, "version")

    assert result["rows"] == 5
    assert {point["label"] for point in result["points"]} == {"a", "b"}
    assert {point["item_id"] for point in result["points"]} == {
        "uuid-0",
        "uuid-1",
        "uuid-2",
        "uuid-3",
        "uuid-4",
    }
    assert all(
        np.isfinite(point["x"]) and np.isfinite(point["y"])
        for point in result["points"]
    )
    assert service.project(frame, request, "version") is result


def test_projection_rejects_invalid_inputs() -> None:
    frame = pd.DataFrame({"__row_id": range(3), "feature_0": [1.0, 2.0, 3.0]})
    service = ProjectionService()

    with pytest.raises(ValueError, match="at least two"):
        service.project(
            frame,
            ProjectionRequest(
                source_id="source", feature_columns=["feature_0"], max_rows=3
            ),
            "version",
        )
    with pytest.raises(ValueError, match="No valid"):
        service.project(
            frame,
            ProjectionRequest(
                source_id="source", feature_columns=["missing"], max_rows=3
            ),
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
            source_id="source",
            method="tsne",
            feature_columns=["feature_0", "feature_1"],
            perplexity=2,
            max_rows=5,
        ),
        "version",
    )

    assert result["rows"] == 5
    assert all(np.isfinite(point["x"]) for point in result["points"])


def test_projection_filters_before_deterministic_sampling() -> None:
    frame = pd.DataFrame(
        {
            "__row_id": range(20),
            "feature_0": np.arange(20, dtype=np.float32),
            "feature_1": np.arange(20, dtype=np.float32) ** 2,
            "group": ["a"] * 10 + ["b"] * 10,
        }
    )
    request = ProjectionRequest(
        source_id="source",
        feature_columns=["feature_0", "feature_1"],
        color_column="group",
        categorical_filters={"group": ["a"]},
        max_rows=5,
    )

    result = ProjectionService().project(frame, request, "version")

    assert result["rows"] == 5
    assert {point["label"] for point in result["points"]} == {"a"}
    assert all(point["row_id"] < 10 for point in result["points"])
    assert [point["row_id"] for point in result["points"]] != list(range(5))


def test_projection_limits_each_class_and_reports_counts() -> None:
    groups = ["a"] * 6 + ["b"] * 5 + [None] * 4
    frame = pd.DataFrame(
        {
            "__row_id": range(15),
            "feature_0": np.arange(15, dtype=np.float32),
            "feature_1": np.arange(15, dtype=np.float32) ** 2,
            "group": groups,
        }
    )
    service = ProjectionService()

    cap_two = service.project(
        frame,
        ProjectionRequest(
            source_id="source",
            feature_columns=["feature_0", "feature_1"],
            color_column="group",
            max_rows=15,
            max_rows_per_class=2,
        ),
        "version",
    )
    cap_three = service.project(
        frame,
        ProjectionRequest(
            source_id="source",
            feature_columns=["feature_0", "feature_1"],
            color_column="group",
            max_rows=15,
            max_rows_per_class=3,
        ),
        "version",
    )

    assert cap_two["available_rows"] == 15
    assert cap_two["rows"] == 6
    assert cap_two["class_counts"] == [
        {"label": "Missing", "available": 4, "displayed": 2},
        {"label": "a", "available": 6, "displayed": 2},
        {"label": "b", "available": 5, "displayed": 2},
    ]
    for label in {"Missing", "a", "b"}:
        selected_two = {
            point["row_id"] for point in cap_two["points"] if point["label"] == label
        }
        selected_three = {
            point["row_id"] for point in cap_three["points"] if point["label"] == label
        }
        assert selected_two < selected_three


def test_projection_applies_individual_class_limits() -> None:
    groups = ["a"] * 6 + ["b"] * 5 + [None] * 4
    frame = pd.DataFrame(
        {
            "__row_id": range(15),
            "feature_0": np.arange(15, dtype=np.float32),
            "feature_1": np.arange(15, dtype=np.float32) ** 2,
            "group": groups,
        }
    )

    result = ProjectionService().project(
        frame,
        ProjectionRequest(
            source_id="source",
            feature_columns=["feature_0", "feature_1"],
            color_column="group",
            max_rows=15,
            max_rows_by_class={"a": 2, "b": 4, "Missing": 1},
        ),
        "version",
    )

    assert result["rows"] == 7
    assert result["class_counts"] == [
        {"label": "Missing", "available": 4, "displayed": 1},
        {"label": "a", "available": 6, "displayed": 2},
        {"label": "b", "available": 5, "displayed": 4},
    ]


def test_projection_individual_limit_overrides_shared_fallback() -> None:
    frame = pd.DataFrame(
        {
            "__row_id": range(12),
            "feature_0": np.arange(12, dtype=np.float32),
            "feature_1": np.arange(12, dtype=np.float32) ** 2,
            "group": ["a"] * 6 + ["b"] * 6,
        }
    )

    result = ProjectionService().project(
        frame,
        ProjectionRequest(
            source_id="source",
            feature_columns=["feature_0", "feature_1"],
            color_column="group",
            max_rows=12,
            max_rows_per_class=2,
            max_rows_by_class={"a": 4},
        ),
        "version",
    )

    assert result["class_counts"] == [
        {"label": "a", "available": 6, "displayed": 4},
        {"label": "b", "available": 6, "displayed": 2},
    ]

    partial = ProjectionService().project(
        frame,
        ProjectionRequest(
            source_id="source",
            feature_columns=["feature_0", "feature_1"],
            color_column="group",
            max_rows=12,
            max_rows_by_class={"a": 2, "absent": 1},
        ),
        "version",
    )
    assert partial["class_counts"] == [
        {"label": "a", "available": 6, "displayed": 2},
        {"label": "b", "available": 6, "displayed": 6},
    ]


def test_projection_class_limit_requires_color_column() -> None:
    frame = pd.DataFrame(
        {
            "__row_id": range(4),
            "feature_0": [0.0, 1.0, 2.0, 3.0],
            "feature_1": [3.0, 2.0, 1.0, 0.0],
        }
    )

    with pytest.raises(ValueError, match="valid Color by"):
        ProjectionService().project(
            frame,
            ProjectionRequest(
                source_id="source",
                feature_columns=["feature_0", "feature_1"],
                max_rows=4,
                max_rows_per_class=2,
            ),
            "version",
        )

    with pytest.raises(ValueError, match="valid Color by"):
        ProjectionService().project(
            frame,
            ProjectionRequest(
                source_id="source",
                feature_columns=["feature_0", "feature_1"],
                max_rows=4,
                max_rows_by_class={"a": 2},
            ),
            "version",
        )


@pytest.mark.parametrize("limit", [0, 50001])
def test_projection_rejects_invalid_individual_class_limits(limit: int) -> None:
    with pytest.raises(ValueError):
        ProjectionRequest(
            source_id="source",
            feature_columns=["feature_0", "feature_1"],
            color_column="group",
            max_rows_by_class={"a": limit},
        )


def test_projection_limits_complete_rows_without_underfilling_classes() -> None:
    groups = ["a"] * 10 + ["b"] * 10
    feature_0 = np.arange(20, dtype=np.float32)
    feature_1 = np.arange(20, dtype=np.float32) ** 2
    feature_0[[1, 3, 5, 7, 9, 11, 13, 15, 17, 19]] = np.nan
    frame = pd.DataFrame(
        {
            "__row_id": range(20),
            "feature_0": feature_0,
            "feature_1": feature_1,
            "group": groups,
        }
    )

    result = ProjectionService().project(
        frame,
        ProjectionRequest(
            source_id="source",
            feature_columns=["feature_0", "feature_1"],
            color_column="group",
            max_rows=20,
            max_rows_by_class={"a": 2, "b": 3},
            random_state=2,
        ),
        "version",
    )

    assert result["available_rows"] == 10
    assert result["rows"] == 5
    assert result["class_counts"] == [
        {"label": "a", "available": 5, "displayed": 2},
        {"label": "b", "available": 5, "displayed": 3},
    ]


def test_projection_global_limit_remains_a_hard_ceiling() -> None:
    frame = pd.DataFrame(
        {
            "__row_id": range(12),
            "feature_0": np.arange(12, dtype=np.float32),
            "feature_1": np.arange(12, dtype=np.float32) ** 2,
            "group": ["a"] * 6 + ["b"] * 6,
        }
    )

    result = ProjectionService().project(
        frame,
        ProjectionRequest(
            source_id="source",
            feature_columns=["feature_0", "feature_1"],
            color_column="group",
            max_rows=5,
            max_rows_by_class={"a": 4, "b": 4},
        ),
        "version",
    )

    assert result["rows"] == 5
    assert all(item["displayed"] <= 4 for item in result["class_counts"])


def test_lda_projection_supports_two_classes() -> None:
    frame = pd.DataFrame(
        {
            "__row_id": range(8),
            "feature_0": [0.0, 0.2, 0.1, 0.3, 4.0, 4.2, 4.1, 4.3],
            "feature_1": [0.1, 0.0, 0.3, 0.2, 4.1, 4.0, 4.3, 4.2],
            "group": ["a"] * 4 + ["b"] * 4,
        }
    )
    result = ProjectionService().project(
        frame,
        ProjectionRequest(
            source_id="source",
            method="lda",
            feature_columns=["feature_0", "feature_1"],
            color_column="group",
            max_rows=8,
            max_rows_per_class=2,
        ),
        "version",
    )

    assert result["rows"] == 4
    assert all(item["displayed"] == 2 for item in result["class_counts"])
    assert all(point["y"] == 0.0 for point in result["points"])


def test_umap_projection() -> None:
    pytest.importorskip("umap")
    frame = pd.DataFrame(
        {
            "__row_id": range(12),
            "feature_0": np.linspace(0, 1, 12),
            "feature_1": np.linspace(1, 0, 12),
        }
    )
    result = ProjectionService().project(
        frame,
        ProjectionRequest(
            source_id="source",
            method="umap",
            feature_columns=["feature_0", "feature_1"],
            umap_neighbors=4,
            max_rows=12,
        ),
        "version",
    )

    assert result["rows"] == 12
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


def test_image_service_does_not_memory_cache_large_previews(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from PIL import Image
    from advanced_visualization.web import images

    path = tmp_path / "source.png"
    Image.new("RGB", (20, 20)).save(path)
    cached_sizes: list[int] = []
    rendered_sizes: list[int] = []
    monkeypatch.setattr(
        images,
        "_thumbnail",
        lambda _path, _modified, _size, max_side: cached_sizes.append(max_side)
        or b"cached",
    )
    monkeypatch.setattr(
        images,
        "_render_image",
        lambda _path, max_side: rendered_sizes.append(max_side) or b"rendered",
    )

    assert images.image_bytes(path, max_side=images.MAX_CACHED_SIDE)[0] == b"cached"
    assert (
        images.image_bytes(path, max_side=images.MAX_CACHED_SIDE + 1)[0] == b"rendered"
    )
    assert cached_sizes == [images.MAX_CACHED_SIDE]
    assert rendered_sizes == [images.MAX_CACHED_SIDE + 1]


def test_image_service_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        image_bytes(tmp_path / "missing.png")


def test_http_routes_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from PIL import Image
    from fastapi import HTTPException
    from advanced_visualization.web.app import app
    from advanced_visualization.web.routes import artifacts as artifact_routes
    from advanced_visualization.web.routes import projections as projection_routes
    from advanced_visualization.web.routes import review as review_routes
    from advanced_visualization.web.routes import sources as source_routes

    image_path = tmp_path / "image.png"
    Image.new("RGB", (30, 20), color=(20, 80, 120)).save(image_path)
    csv_path = tmp_path / "source.csv"
    pd.DataFrame(
        {
            "id": ["a", "b", "c"],
            "label": [0, 1, 1],
            "score": [0.9, 0.8, 0.2],
            "image_path": [str(image_path)] * 3,
            "feature_0": [0.0, 1.0, 2.0],
            "feature_1": [2.0, 1.0, 0.0],
        }
    ).to_csv(csv_path, index=False)
    artifact_dir = tmp_path / "artifact"
    gradcam_dir = artifact_dir / "gradcam"
    gradcam_dir.mkdir(parents=True)
    from advanced_visualization.core.images import image_cache_digest

    digest = image_cache_digest(image_path)
    Image.new("RGB", (30, 20), color=(180, 40, 30)).save(
        gradcam_dir / f"{digest}_gradcam_logit.png"
    )
    Image.new("RGB", (30, 20), color=(30, 180, 40)).save(
        gradcam_dir / f"{digest}_gradcampp_logit.png"
    )
    fixed = FixedRepository(
        DataSource("source", "Source", csv_path, "model", artifact_dir)
    )
    for route_module in (
        artifact_routes,
        projection_routes,
        review_routes,
        source_routes,
    ):
        monkeypatch.setattr(route_module, "repository", fixed)

    endpoints = {
        route.name: route.endpoint for route in app.routes if hasattr(route, "endpoint")
    }

    assert endpoints["health"]() == {"status": "ok"}
    assert endpoints["favicon"]().status_code == 204
    assert source_routes.sources()[0].id == "source"
    schema_response = source_routes.schema("source")
    assert schema_response.feature_columns == ["feature_0", "feature_1"]
    assert set(schema_response.prepared_gradcam_methods) == {"gradcam", "gradcam++"}

    review_response = review_routes.review(
        FilterRequest(
            source_id="source",
            item_id_column="id",
            image_column="image_path",
            truth_column="label",
            prediction_column="score",
            page_size=12,
        )
    )
    first_row = review_response["rows"][0]
    assert first_row["image_url"].endswith("max_side=480")
    assert "/api/gradcam/" in first_row["gradcam_url"]
    image_response = artifact_routes.image(
        "source", first_row["row_id"], "image_path", 480
    )
    assert image_response.media_type == "image/jpeg"

    with pytest.raises(HTTPException) as missing:
        artifact_routes.image("source", 99, "image_path", 480)
    assert missing.value.status_code == 404

    detail = review_routes.point_detail(
        "source", first_row["row_id"], "image_path", "", "id", "score", "label"
    )
    assert detail["values"]["id"] in {"a", "b", "c"}
    assert "/api/gradcam/" in detail["gradcam_url"]
    assert "method=gradcam%2B%2B" in detail["gradcam_plus_url"]
    cam_response = artifact_routes.prepared_gradcam(
        "source", first_row["row_id"], "image_path", "gradcam", 200
    )
    assert cam_response.media_type == "image/jpeg"

    projection_response = projection_routes.projection(
        ProjectionRequest(
            source_id="source",
            feature_columns=["feature_0", "feature_1"],
            max_rows=3,
        )
    )
    assert projection_response["rows"] == 3
