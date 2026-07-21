"""Pure dataframe filtering and paging logic."""
from __future__ import annotations

import math

import pandas as pd

from advanced_visualization.core.evaluation import attach_binary_evaluation, evaluate_binary
from advanced_visualization.web.models import FilterRequest


INTERNAL_COLUMNS = {
    "__has_eval", "__prediction_score", "__predicted_positive",
    "__actual_positive", "__truth_is_valid", "__truth_is_invalid",
    "__is_failure", "__failure_type", "__confidence",
}


def _known_column(frame: pd.DataFrame, value: str) -> str | None:
    return value if value and value in frame.columns and value not in INTERNAL_COLUMNS else None


def enrich_failures(frame: pd.DataFrame, request: FilterRequest) -> pd.DataFrame:
    truth_column = _known_column(frame, request.truth_column)
    prediction_column = _known_column(frame, request.prediction_column)
    return attach_binary_evaluation(
        frame,
        evaluate_binary(frame, truth_column, prediction_column, request.threshold),
    )


def _apply_truth_filter(frame: pd.DataFrame, mode: str) -> pd.DataFrame:
    if mode == "valid":
        return frame[frame["__truth_is_valid"]]
    if mode == "positive":
        return frame[frame["__actual_positive"] & frame["__truth_is_valid"]]
    if mode == "negative":
        return frame[~frame["__actual_positive"] & frame["__truth_is_valid"]]
    if mode == "invalid":
        return frame[frame["__truth_is_invalid"]]
    return frame


def _apply_failure_filter(frame: pd.DataFrame, request: FilterRequest) -> pd.DataFrame:
    mode = request.failure_mode
    if mode == "failures":
        return frame[frame["__is_failure"]]
    if mode == "correct":
        return frame[frame["__failure_type"].eq("correct")]
    if mode == "false_positives":
        return frame[frame["__failure_type"].eq("false positive")]
    if mode == "false_negatives":
        return frame[frame["__failure_type"].eq("false negative")]
    if mode == "high_confidence":
        return frame[frame["__is_failure"] & frame["__confidence"].ge(request.high_confidence)]
    if mode == "low_confidence":
        return frame[frame["__is_failure"] & frame["__confidence"].le(request.low_confidence)]
    return frame


def filter_frame(frame: pd.DataFrame, request: FilterRequest) -> pd.DataFrame:
    result = _apply_truth_filter(enrich_failures(frame, request), request.truth_rows)
    result = _apply_failure_filter(result, request)
    search_columns = [column for column in request.search_columns if _known_column(result, column)]
    if request.search.strip() and search_columns:
        mask = pd.Series(False, index=result.index)
        for column in search_columns:
            mask |= result[column].astype(str).str.contains(request.search.strip(), case=False, na=False, regex=False)
        result = result[mask]
    for column, values in request.categorical_filters.items():
        if not _known_column(result, column):
            continue
        if not values:
            return result.iloc[0:0]
        result = result[result[column].fillna("Missing").astype(str).isin(values)]

    sort_column, ascending = {
        "confidence_desc": ("__confidence", False),
        "confidence_asc": ("__confidence", True),
        "prediction_desc": ("__prediction_score", False),
        "prediction_asc": ("__prediction_score", True),
        "row_order": ("__row_id", True),
    }[request.sort]
    return result.sort_values(sort_column, ascending=ascending, na_position="last", kind="stable")


def page_frame(frame: pd.DataFrame, request: FilterRequest) -> tuple[pd.DataFrame, dict]:
    total = len(frame)
    pages = max(1, math.ceil(total / request.page_size))
    page = min(request.page, pages)
    start = (page - 1) * request.page_size
    scored = frame[frame["__has_eval"]]
    failures = int(frame["__is_failure"].sum())
    metrics = {
        "rows": total,
        "scored": len(scored),
        "failures": failures,
        "failure_rate": failures / len(scored) if len(scored) else 0.0,
        "high_confidence_failures": int((frame["__is_failure"] & frame["__confidence"].ge(request.high_confidence)).sum()),
        "low_confidence_failures": int((frame["__is_failure"] & frame["__confidence"].le(request.low_confidence)).sum()),
    }
    return frame.iloc[start:start + request.page_size], {
        "total": total, "page": page, "pages": pages, "page_size": request.page_size, "metrics": metrics,
    }
