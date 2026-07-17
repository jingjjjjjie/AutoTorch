"""Pure dataframe filtering and paging logic."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from advanced_visualization.web.models import FilterRequest


INTERNAL_COLUMNS = {
    "__has_eval", "__prediction_score", "__predicted_positive",
    "__actual_positive", "__truth_is_valid", "__truth_is_invalid",
    "__is_failure", "__failure_type", "__confidence",
}


def _known_column(frame: pd.DataFrame, value: str) -> str | None:
    return value if value and value in frame.columns and value not in INTERNAL_COLUMNS else None


def enrich_failures(frame: pd.DataFrame, request: FilterRequest) -> pd.DataFrame:
    result = frame.copy()
    truth_column = _known_column(result, request.truth_column)
    prediction_column = _known_column(result, request.prediction_column)
    if not truth_column or not prediction_column:
        result["__has_eval"] = False
        result["__prediction_score"] = np.nan
        result["__actual_positive"] = False
        result["__truth_is_valid"] = False
        result["__truth_is_invalid"] = False
        result["__is_failure"] = False
        result["__failure_type"] = "unscored"
        result["__confidence"] = np.nan
        return result

    score = pd.to_numeric(result[prediction_column], errors="coerce")
    truth = pd.to_numeric(result[truth_column], errors="coerce")
    predicted_positive = score.ge(request.threshold)
    actual_positive = truth.eq(1)
    truth_valid = truth.isin([0, 1])
    has_eval = score.notna() & truth_valid
    failure = predicted_positive.ne(actual_positive) & has_eval
    result["__has_eval"] = has_eval
    result["__prediction_score"] = score
    result["__actual_positive"] = actual_positive
    result["__truth_is_valid"] = truth_valid
    result["__truth_is_invalid"] = truth.eq(-1)
    result["__is_failure"] = failure
    result["__failure_type"] = np.select(
        [~has_eval, predicted_positive & ~actual_positive, ~predicted_positive & actual_positive, ~failure],
        ["unscored", "false positive", "false negative", "correct"],
        default="failure",
    )
    result["__confidence"] = np.maximum(score, 1.0 - score)
    return result


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

