"""Notebook-equivalent experiment analysis built from shared evaluation logic."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from advanced_visualization.core.evaluation import BinaryEvaluation, evaluate_binary, evaluation_metrics


ANALYSIS_OUTCOMES = (
    "all",
    "ori_wrong",
    "crop_wrong",
    "merged_wrong",
    "ori_only_correct",
    "crop_only_correct",
    "merged_fixes_both",
    "merged_regression",
    "ori_crop_disagree",
)


@dataclass(frozen=True)
class AnalysisResult:
    frame: pd.DataFrame
    evaluations: dict[str, BinaryEvaluation]

    def summary(self) -> dict[str, dict[str, int | float]]:
        return {name: evaluation_metrics(evaluation) for name, evaluation in self.evaluations.items()}


def normalize_quality_values(series: pd.Series) -> pd.Series:
    """Normalize the quality labels used by the results-analysis notebook."""
    mapped = series.mask(series.eq("No Quality Issue"), 0)
    mapped = mapped.mask(series.eq("Quality Issue"), 1)
    return pd.to_numeric(mapped, errors="coerce")


def filter_analysis_frame(
    frame: pd.DataFrame,
    *,
    quality_column: str = "",
    quality_mode: str = "all",
    subclass_column: str = "",
    exclude_unknown_subclass: bool = True,
    identity_column: str = "",
    excluded_identities: tuple[str, ...] = (),
    categorical_filters: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """Apply the notebook's data cleanup plus user-selected filters."""
    filtered = frame.copy()
    if quality_column and quality_column in filtered:
        quality = normalize_quality_values(filtered[quality_column])
        if quality_mode == "high_quality":
            filtered = filtered[quality.eq(0)]
        elif quality_mode == "quality_issue":
            filtered = filtered[quality.eq(1)]
        elif quality_mode == "known":
            filtered = filtered[quality.notna()]
    if exclude_unknown_subclass and subclass_column and subclass_column in filtered:
        filtered = filtered[filtered[subclass_column].fillna("").astype(str).str.casefold().ne("unknown")]
    if excluded_identities and identity_column and identity_column in filtered:
        filtered = filtered[~filtered[identity_column].fillna("Missing").astype(str).isin(excluded_identities)]
    for column, values in (categorical_filters or {}).items():
        if column not in filtered:
            continue
        if not values:
            return filtered.iloc[0:0]
        filtered = filtered[filtered[column].fillna("Missing").astype(str).isin(values)]
    return filtered


def analyze_ori_crop(
    frame: pd.DataFrame,
    *,
    truth_column: str,
    ori_prediction_column: str,
    crop_prediction_column: str,
    threshold: float = 0.5,
) -> AnalysisResult:
    """Evaluate ori, crop, and their arithmetic-mean merged score."""
    missing = [
        column for column in (truth_column, ori_prediction_column, crop_prediction_column)
        if not column or column not in frame
    ]
    if missing:
        raise ValueError(f"Analysis is missing required column(s): {missing}")
    result = frame.copy()
    result["__ori_score"] = pd.to_numeric(result[ori_prediction_column], errors="coerce")
    result["__crop_score"] = pd.to_numeric(result[crop_prediction_column], errors="coerce")
    result["__merged_score"] = (result["__ori_score"] + result["__crop_score"]) / 2.0
    evaluations = {
        # The source notebook classifies scores with ``score > threshold``.
        "ori": evaluate_binary(result, truth_column, "__ori_score", threshold, inclusive=False),
        "crop": evaluate_binary(result, truth_column, "__crop_score", threshold, inclusive=False),
        "merged": evaluate_binary(result, truth_column, "__merged_score", threshold, inclusive=False),
    }
    for name, evaluation in evaluations.items():
        result[f"__{name}_failure_type"] = evaluation.failure_type
        result[f"__{name}_correct"] = evaluation.has_eval & ~evaluation.is_failure
    result["__ori_crop_disagree"] = (
        evaluations["ori"].has_eval
        & evaluations["crop"].has_eval
        & evaluations["ori"].predicted_positive.ne(evaluations["crop"].predicted_positive)
    )
    result["__analysis_outcome"] = np.select(
        [
            result["__merged_correct"] & ~result["__ori_correct"] & ~result["__crop_correct"],
            ~result["__merged_correct"] & (result["__ori_correct"] | result["__crop_correct"]),
            result["__ori_correct"] & ~result["__crop_correct"],
            result["__crop_correct"] & ~result["__ori_correct"],
            ~result["__merged_correct"],
        ],
        [
            "merged_fixes_both",
            "merged_regression",
            "ori_only_correct",
            "crop_only_correct",
            "merged_wrong",
        ],
        default="all_correct",
    )
    return AnalysisResult(result, evaluations)


def analysis_outcome_mask(result: AnalysisResult, outcome: str) -> pd.Series:
    frame = result.frame
    if outcome == "all":
        return result.evaluations["merged"].has_eval
    if outcome == "ori_wrong":
        return result.evaluations["ori"].has_eval & result.evaluations["ori"].is_failure
    if outcome == "crop_wrong":
        return result.evaluations["crop"].has_eval & result.evaluations["crop"].is_failure
    if outcome == "merged_wrong":
        return result.evaluations["merged"].has_eval & result.evaluations["merged"].is_failure
    if outcome == "ori_crop_disagree":
        return frame["__ori_crop_disagree"]
    if outcome in {"ori_only_correct", "crop_only_correct", "merged_fixes_both", "merged_regression"}:
        return frame["__analysis_outcome"].eq(outcome)
    raise ValueError(f"Unknown analysis outcome: {outcome}")


def grouped_analysis(result: AnalysisResult, group_columns: tuple[str, ...]) -> list[dict]:
    """Return notebook-style metrics for each requested group combination."""
    columns = [column for column in group_columns if column and column in result.frame]
    if not columns:
        return []
    rows: list[dict[str, object]] = []
    grouper = columns[0] if len(columns) == 1 else columns
    evaluable = result.frame[result.evaluations["merged"].has_eval]
    for key, group in evaluable.groupby(grouper, dropna=False, sort=True):
        values = key if isinstance(key, tuple) else (key,)
        row: dict[str, object] = {
            column: ("Missing" if pd.isna(value) else str(value))
            for column, value in zip(columns, values)
        }
        row["count"] = int(len(group))
        mask = result.frame.index.isin(group.index)
        mask_series = pd.Series(mask, index=result.frame.index)
        for name, evaluation in result.evaluations.items():
            metrics = evaluation_metrics(evaluation, mask_series)
            row[f"{name}_error_rate"] = 1.0 - float(metrics["accuracy"])
            row[f"{name}_errors"] = int(metrics["failures"])
            row[f"{name}_apcer"] = float(metrics["apcer"])
            row[f"{name}_bpcer"] = float(metrics["bpcer"])
        rows.append(row)
    return rows
