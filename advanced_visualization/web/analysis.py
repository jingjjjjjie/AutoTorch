"""Notebook-equivalent results-analysis application service."""

from __future__ import annotations

import math

import numpy as np
from advanced_visualization.core.analysis import (
    analysis_outcome_mask,
    analyze_ori_crop,
    filter_analysis_frame,
    grouped_analysis,
)
from advanced_visualization.web.artifacts import image_url
from advanced_visualization.web.models import AnalysisRequest
from advanced_visualization.web.repository import DatasetRepository
from advanced_visualization.web.serialization import json_value


def analysis_page(repository: DatasetRepository, request: AnalysisRequest) -> dict:
    source = repository.source(request.source_id)
    frame = repository.review_dataframe(request.source_id)
    if request.item_id_column not in frame:
        raise ValueError(
            f"Analysis is missing item ID column: {request.item_id_column}"
        )
    filtered = filter_analysis_frame(
        frame,
        quality_column=request.quality_column,
        quality_mode=request.quality_mode,
        subclass_column=request.subclass_column,
        exclude_unknown_subclass=request.exclude_unknown_subclass,
        identity_column=request.identity_column,
        excluded_identities=tuple(request.excluded_identities),
        categorical_filters=request.categorical_filters,
    )
    result = analyze_ori_crop(
        filtered,
        truth_column=request.truth_column,
        ori_prediction_column=request.ori_prediction_column,
        crop_prediction_column=request.crop_prediction_column,
        threshold=request.threshold,
    )
    selected = analysis_outcome_mask(result, request.outcome)
    if request.search.strip():
        selected &= (
            result.frame[request.item_id_column]
            .astype(str)
            .str.contains(request.search.strip(), case=False, na=False, regex=False)
        )
    working = result.frame[selected].copy()
    working["__ori_crop_delta"] = working["__ori_score"] - working["__crop_score"]
    working["__merged_confidence"] = np.maximum(
        working["__merged_score"], 1.0 - working["__merged_score"]
    )
    sort_column, ascending = {
        "merged_confidence_desc": ("__merged_confidence", False),
        "ori_crop_delta_desc": ("__ori_crop_delta", False),
        "ori_crop_delta_asc": ("__ori_crop_delta", True),
        "item_id": (request.item_id_column, True),
    }[request.sort]
    working = working.sort_values(
        sort_column, ascending=ascending, na_position="last", kind="stable"
    )
    total = len(working)
    pages = max(1, math.ceil(total / request.page_size))
    page_number = min(request.page, pages)
    page = working.iloc[
        (page_number - 1) * request.page_size : page_number * request.page_size
    ]
    rows = []
    for _, row in page.iterrows():
        row_id = int(row["__row_id"])
        rows.append(
            {
                "row_id": row_id,
                "item_id": str(row.get(request.item_id_column, row_id)),
                "truth": json_value(row[request.truth_column]),
                "ori_score": json_value(row["__ori_score"]),
                "crop_score": json_value(row["__crop_score"]),
                "merged_score": json_value(row["__merged_score"]),
                "ori_failure_type": str(row["__ori_failure_type"]),
                "crop_failure_type": str(row["__crop_failure_type"]),
                "merged_failure_type": str(row["__merged_failure_type"]),
                "outcome": str(row["__analysis_outcome"]),
                "subclass": (
                    json_value(row.get(request.subclass_column))
                    if request.subclass_column
                    else None
                ),
                "identity": (
                    json_value(row.get(request.identity_column))
                    if request.identity_column
                    else None
                ),
                "quality": (
                    json_value(row.get(request.quality_column))
                    if request.quality_column
                    else None
                ),
                "original_image_url": image_url(
                    source.id, row_id, request.original_image_column
                ),
                "crop_image_url": image_url(
                    source.id, row_id, request.crop_image_column
                ),
            }
        )
    breakdowns = {
        "subclass": grouped_analysis(result, (request.subclass_column,)),
        "identity": grouped_analysis(result, (request.identity_column,)),
        "identity_subclass": grouped_analysis(
            result, (request.identity_column, request.subclass_column)
        ),
    }
    return {
        "total": total,
        "filtered_rows": len(filtered),
        "page": page_number,
        "pages": pages,
        "page_size": request.page_size,
        "summary": result.summary(),
        "breakdowns": breakdowns,
        "rows": rows,
    }
