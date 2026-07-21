"""Behavioral tests for model comparison and notebook-style analysis."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from advanced_visualization.core.analysis import (
    analyze_ori_crop,
    filter_analysis_frame,
    grouped_analysis,
)
from advanced_visualization.core.comparison import compare_frames
from advanced_visualization.web.analysis import analysis_page
from advanced_visualization.web.comparisons import comparison_page
from advanced_visualization.web.models import AnalysisRequest, ComparisonRequest
from advanced_visualization.web.repository import DataSource, DatasetRepository


def _with_row_ids(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result.insert(0, "__row_id", range(len(result)))
    return result


def test_comparison_aligns_by_id_and_classifies_every_outcome() -> None:
    frame_a = _with_row_ids(pd.DataFrame({
        "id": ["both", "a-wins", "b-wins", "both-wrong", "mismatch", "only-a", "unscored"],
        "label": [1, 1, 0, 0, 1, 1, 0],
        "score": [0.9, 0.9, 0.9, 0.9, 0.7, 0.8, None],
    }))
    frame_b = _with_row_ids(pd.DataFrame({
        "id": ["only-b", "unscored", "mismatch", "both-wrong", "b-wins", "a-wins", "both"],
        "label": [0, 0, 0, 0, 0, 1, 1],
        "score": [0.2, 0.1, 0.7, 0.8, 0.1, 0.1, 0.8],
    }))

    result = compare_frames(
        frame_a,
        frame_b,
        item_id_column_a="id",
        item_id_column_b="id",
        truth_column_a="label",
        truth_column_b="label",
        prediction_column_a="score",
        prediction_column_b="score",
    )
    outcomes = result.frame.set_index("__comparison_id")["__comparison_outcome"].to_dict()

    assert outcomes == {
        "both": "both_correct",
        "a-wins": "a_only_correct",
        "b-wins": "b_only_correct",
        "both-wrong": "both_wrong",
        "mismatch": "truth_mismatch",
        "only-a": "only_in_a",
        "unscored": "unscored",
        "only-b": "only_in_b",
    }
    assert result.alignment == {
        "source_a_rows": 7,
        "source_b_rows": 7,
        "matched": 6,
        "only_in_a": 1,
        "only_in_b": 1,
        "missing_id_a": 0,
        "missing_id_b": 0,
        "truth_mismatches": 1,
        "duplicate_ids_a": 0,
        "duplicate_ids_b": 0,
    }
    assert result.summary()["a"]["accuracy"] == pytest.approx(0.5)
    assert result.summary()["b"]["accuracy"] == pytest.approx(0.5)


def test_comparison_aligns_duplicate_ids_by_occurrence_without_cartesian_join() -> None:
    frame_a = _with_row_ids(pd.DataFrame({
        "id": ["same", "same"], "label": [0, 1], "score": [0.1, 0.9],
    }))
    frame_b = _with_row_ids(pd.DataFrame({
        "id": ["same", "same"], "label": [0, 1], "score": [0.2, 0.8],
    }))

    result = compare_frames(
        frame_a,
        frame_b,
        item_id_column_a="id",
        item_id_column_b="id",
        truth_column_a="label",
        truth_column_b="label",
        prediction_column_a="score",
        prediction_column_b="score",
    )

    assert len(result.frame) == 2
    assert result.frame["__comparison_occurrence"].tolist() == [0, 1]
    assert result.alignment["duplicate_ids_a"] == 1
    assert result.alignment["duplicate_ids_b"] == 1


def test_analysis_matches_notebook_strict_threshold_and_metrics() -> None:
    frame = _with_row_ids(pd.DataFrame({
        "id": list("abcde"),
        "label": [1, 1, 0, 0, -1],
        "ori": [0.5, 0.8, 0.5, 0.7, 0.9],
        "crop": [0.5, 0.2, 0.5, 0.1, 0.2],
        "subclass": ["print", "print", "genuine", "genuine", "unknown-label"],
    }))

    result = analyze_ori_crop(
        frame,
        truth_column="label",
        ori_prediction_column="ori",
        crop_prediction_column="crop",
        threshold=0.5,
    )

    # The notebook uses score > threshold, so exactly 0.5 predicts genuine.
    assert result.evaluations["ori"].predicted_positive.tolist() == [False, True, False, True, True]
    assert result.summary()["ori"] == {
        "rows": 4,
        "correct": 2,
        "failures": 2,
        "accuracy": 0.5,
        "apcer": 0.5,
        "apcer_errors": 1,
        "attack_rows": 2,
        "bpcer": 0.5,
        "bpcer_errors": 1,
        "genuine_rows": 2,
    }
    assert result.summary()["crop"]["apcer"] == 1.0
    assert result.summary()["crop"]["bpcer"] == 0.0
    breakdown = grouped_analysis(result, ("subclass",))
    assert {row["subclass"]: row["count"] for row in breakdown} == {
        "genuine": 2,
        "print": 2,
    }


def test_analysis_cleanup_matches_notebook_quality_and_exclusions() -> None:
    frame = pd.DataFrame({
        "quality": ["No Quality Issue", "Quality Issue", None, 0, 1],
        "subclass": ["Print", "Unknown", "Replay", "Print", "Replay"],
        "identity": ["keep", "keep", "keep", "exclude", "keep"],
    })

    filtered = filter_analysis_frame(
        frame,
        quality_column="quality",
        quality_mode="known",
        subclass_column="subclass",
        identity_column="identity",
        excluded_identities=("exclude",),
    )

    assert filtered.index.tolist() == [0, 4]


class MultiSourceRepository(DatasetRepository):
    def __init__(self, sources: list[DataSource]) -> None:
        super().__init__()
        self._sources = sources

    def sources(self) -> list[DataSource]:
        return self._sources


def test_comparison_and_analysis_services_return_page_contracts(tmp_path: Path) -> None:
    base = pd.DataFrame({
        "id": [f"id-{index}" for index in range(12)],
        "label": [0, 1] * 6,
        "ori": [0.1, 0.9] * 6,
        "crop": [0.2, 0.8] * 6,
        "group": ["genuine", "attack"] * 6,
        "identity": ["month-a"] * 12,
        "quality": ["No Quality Issue"] * 12,
    })
    path_a = tmp_path / "a.csv"
    path_b = tmp_path / "b.csv"
    base.to_csv(path_a, index=False)
    base.assign(ori=[0.9, 0.1] * 6).to_csv(path_b, index=False)
    repository = MultiSourceRepository([
        DataSource("a", "Experiment A", path_a, "a", None),
        DataSource("b", "Experiment B", path_b, "b", None),
    ])

    comparison = comparison_page(repository, ComparisonRequest(
        source_a_id="a",
        source_b_id="b",
        item_id_column_a="id",
        item_id_column_b="id",
        truth_column_a="label",
        truth_column_b="label",
        prediction_column_a="ori",
        prediction_column_b="ori",
        subclass_column="group",
        page_size=12,
    ))
    analysis = analysis_page(repository, AnalysisRequest(
        source_id="a",
        item_id_column="id",
        truth_column="label",
        ori_prediction_column="ori",
        crop_prediction_column="crop",
        subclass_column="group",
        identity_column="identity",
        quality_column="quality",
        quality_mode="known",
        page_size=12,
    ))

    assert comparison["summary"]["outcomes"]["a_only_correct"] == 12
    assert len(comparison["rows"]) == 12
    assert analysis["summary"]["merged"]["accuracy"] == 1.0
    assert analysis["breakdowns"]["subclass"][0]["count"] == 6
    assert len(analysis["rows"]) == 12
