"""Model-comparison application service for the web API."""
from __future__ import annotations

import math

import pandas as pd

from advanced_visualization.core.comparison import compare_frames
from advanced_visualization.web.artifacts import gradcam_url, image_url
from advanced_visualization.web.models import ComparisonRequest
from advanced_visualization.web.repository import DatasetRepository
from advanced_visualization.web.serialization import json_value


def _row_id(value) -> int | None:
    return None if pd.isna(value) else int(value)


def comparison_page(repository: DatasetRepository, request: ComparisonRequest) -> dict:
    source_a = repository.source(request.source_a_id)
    source_b = repository.source(request.source_b_id)
    frame_a = repository.review_dataframe(request.source_a_id)
    frame_b = repository.review_dataframe(request.source_b_id)
    metadata: tuple[str, ...] = tuple(dict.fromkeys(filter(None, [
        request.subclass_column,
        request.identity_column,
        request.quality_column,
        request.image_column_a,
        request.image_column_b,
        request.gradcam_column_a,
        request.gradcam_column_b,
        *request.categorical_filters.keys(),
    ])))
    result = compare_frames(
        frame_a,
        frame_b,
        item_id_column_a=request.item_id_column_a,
        item_id_column_b=request.item_id_column_b,
        truth_column_a=request.truth_column_a,
        truth_column_b=request.truth_column_b,
        prediction_column_a=request.prediction_column_a,
        prediction_column_b=request.prediction_column_b,
        threshold_a=request.threshold_a,
        threshold_b=request.threshold_b,
        metadata_columns_a=metadata,
        metadata_columns_b=metadata,
    )
    base_selected = pd.Series(True, index=result.frame.index, dtype=bool)
    for column, values in request.categorical_filters.items():
        merged_column = f"a__{column}" if f"a__{column}" in result.frame else f"b__{column}"
        if merged_column not in result.frame:
            continue
        if not values:
            base_selected &= False
            continue
        base_selected &= result.frame[merged_column].fillna("Missing").astype(str).isin(values)
    if request.search.strip():
        base_selected &= result.frame["__comparison_id"].str.contains(
            request.search.strip(), case=False, na=False, regex=False
        )
    selected = base_selected.copy()
    if request.outcomes:
        selected &= result.frame["__comparison_outcome"].isin(request.outcomes)
    working = result.frame[selected].copy()
    sort_column, ascending = {
        "true_confidence_delta_desc": ("__true_confidence_delta", False),
        "true_confidence_delta_asc": ("__true_confidence_delta", True),
        "score_delta_desc": ("__score_delta", False),
        "score_delta_asc": ("__score_delta", True),
        "item_id": ("__comparison_id", True),
    }[request.sort]
    working = working.sort_values(sort_column, ascending=ascending, na_position="last", kind="stable")
    total = len(working)
    pages = max(1, math.ceil(total / request.page_size))
    page_number = min(request.page, pages)
    page = working.iloc[(page_number - 1) * request.page_size:page_number * request.page_size]

    rows = []
    for _, row in page.iterrows():
        row_id_a = _row_id(row.get("a____row_id"))
        row_id_b = _row_id(row.get("b____row_id"))
        source_row_a = frame_a.iloc[row_id_a] if row_id_a is not None else pd.Series(dtype=object)
        source_row_b = frame_b.iloc[row_id_b] if row_id_b is not None else pd.Series(dtype=object)
        metadata_values = {}
        for column in (request.subclass_column, request.identity_column, request.quality_column):
            if not column:
                continue
            value = row.get(f"a__{column}", row.get(f"b__{column}"))
            metadata_values[column] = json_value(value)
        rows.append({
            "item_id": str(row["__comparison_id"]),
            "occurrence": int(row["__comparison_occurrence"]),
            "row_id_a": row_id_a,
            "row_id_b": row_id_b,
            "truth": json_value(row["__truth"]),
            "outcome": str(row["__comparison_outcome"]),
            "transition": str(row["__transition"]),
            "a_score": json_value(row["__a_score"]),
            "b_score": json_value(row["__b_score"]),
            "score_delta": json_value(row["__score_delta"]),
            "true_confidence_delta": json_value(row["__true_confidence_delta"]),
            "a_failure_type": str(row["__a_failure_type"]),
            "b_failure_type": str(row["__b_failure_type"]),
            "metadata": metadata_values,
            "a_image_url": image_url(source_a.id, row_id_a, request.image_column_a),
            "b_image_url": image_url(source_b.id, row_id_b, request.image_column_b),
            "a_gradcam_url": gradcam_url(
                source_a, source_row_a, row_id_a,
                image_column=request.image_column_a,
                gradcam_column=request.gradcam_column_a,
                method=request.gradcam_method,
                target=request.gradcam_target,
                layer=request.gradcam_layer_a,
            ),
            "b_gradcam_url": gradcam_url(
                source_b, source_row_b, row_id_b,
                image_column=request.image_column_b,
                gradcam_column=request.gradcam_column_b,
                method=request.gradcam_method,
                target=request.gradcam_target,
                layer=request.gradcam_layer_b,
            ),
            "a_genuine_gradcam_url": gradcam_url(
                source_a, source_row_a, row_id_a,
                image_column=request.image_column_a,
                gradcam_column=request.gradcam_column_a,
                method=request.gradcam_method,
                target="genuine",
                layer=request.gradcam_layer_a,
            ),
            "a_fraud_gradcam_url": gradcam_url(
                source_a, source_row_a, row_id_a,
                image_column=request.image_column_a,
                gradcam_column=request.gradcam_column_a,
                method=request.gradcam_method,
                target="fraud",
                layer=request.gradcam_layer_a,
            ),
            "b_genuine_gradcam_url": gradcam_url(
                source_b, source_row_b, row_id_b,
                image_column=request.image_column_b,
                gradcam_column=request.gradcam_column_b,
                method=request.gradcam_method,
                target="genuine",
                layer=request.gradcam_layer_b,
            ),
            "b_fraud_gradcam_url": gradcam_url(
                source_b, source_row_b, row_id_b,
                image_column=request.image_column_b,
                gradcam_column=request.gradcam_column_b,
                method=request.gradcam_method,
                target="fraud",
                layer=request.gradcam_layer_b,
            ),
        })
    return {
        "total": total,
        "page": page_number,
        "pages": pages,
        "page_size": request.page_size,
        "summary": result.summary(base_selected),
        "rows": rows,
    }
