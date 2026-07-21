"""Image-review and point-inspection routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from advanced_visualization.web.artifacts import (
    image_url,
    prepared_gradcam_path,
    prepared_gradcam_url,
)
from advanced_visualization.web.filtering import filter_frame, page_frame
from advanced_visualization.web.models import FilterRequest, PageResponse
from advanced_visualization.web.repository import repository
from advanced_visualization.web.serialization import json_value


router = APIRouter(prefix="/api", tags=["review"])


@router.post("/review", response_model=PageResponse)
def review(request: FilterRequest) -> dict:
    try:
        source = repository.source(request.source_id)
        frame = repository.review_dataframe(request.source_id)
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    requested_columns = {
        "__row_id",
        request.item_id_column,
        request.subclass_column,
        request.truth_column,
        request.prediction_column,
        request.image_column,
        request.gradcam_column,
        *request.search_columns,
        *request.categorical_filters.keys(),
    }
    requested_columns.discard("")
    review_frame = frame.loc[
        :, [column for column in frame.columns if column in requested_columns]
    ]
    filtered = filter_frame(review_frame, request)
    page, metadata = page_frame(filtered, request)
    visible_columns = list(
        dict.fromkeys(
            filter(
                None,
                [
                    request.item_id_column,
                    request.subclass_column,
                    request.truth_column,
                    request.prediction_column,
                    "__failure_type",
                    "__confidence",
                ],
            )
        )
    )
    rows = []
    for _, row in page.iterrows():
        row_id = int(row["__row_id"])
        cam_url = (
            image_url(request.source_id, row_id, request.gradcam_column)
            if request.gradcam_target == "fraud"
            else ""
        )
        if not cam_url and request.image_column:
            path = prepared_gradcam_path(
                source,
                row,
                request.image_column,
                request.gradcam_method,
                request.gradcam_target,
            )
            if path is not None:
                cam_url = prepared_gradcam_url(
                    request.source_id,
                    row_id,
                    request.image_column,
                    request.gradcam_method,
                    request.gradcam_target,
                )
        rows.append(
            {
                "row_id": row_id,
                "values": {
                    column: json_value(row[column])
                    for column in visible_columns
                    if column in row.index
                },
                "image_url": image_url(request.source_id, row_id, request.image_column),
                "gradcam_url": cam_url,
            }
        )
    return metadata | {"rows": rows}


@router.get("/points/{source_id}/{row_id}")
def point_detail(
    source_id: str,
    row_id: int,
    image_column: str = "",
    gradcam_column: str = "",
    item_id_column: str = "",
    prediction_column: str = "",
    group_column: str = "",
    gradcam_target: str = "fraud",
) -> dict:
    try:
        source = repository.source(source_id)
        frame = repository.review_dataframe(source_id)
        schema_details = repository.schema(source_id)
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if row_id < 0 or row_id >= len(frame):
        raise HTTPException(status_code=404, detail="Row does not exist.")
    if gradcam_target not in {"fraud", "genuine"}:
        raise HTTPException(
            status_code=400, detail="CAM target must be fraud or genuine."
        )
    row = frame.iloc[row_id]
    allowed_images = set(schema_details["image_columns"])
    image_column = image_column if image_column in allowed_images else ""
    gradcam_column = gradcam_column if gradcam_column in allowed_images else ""
    values = {
        column: json_value(row[column])
        for column in (item_id_column, group_column, prediction_column)
        if column and column in row.index
    }

    cam_url = (
        image_url(source_id, row_id, gradcam_column, max_side=900)
        if gradcam_column and gradcam_target == "fraud"
        else ""
    )
    gradcam_plus_url = ""
    if image_column:
        if not cam_url and prepared_gradcam_path(
            source, row, image_column, "gradcam", gradcam_target
        ):
            cam_url = prepared_gradcam_url(
                source_id, row_id, image_column, "gradcam", gradcam_target
            )
        if prepared_gradcam_path(
            source, row, image_column, "gradcam++", gradcam_target
        ):
            gradcam_plus_url = prepared_gradcam_url(
                source_id, row_id, image_column, "gradcam++", gradcam_target
            )
    return {
        "row_id": row_id,
        "values": values,
        "image_url": image_url(source_id, row_id, image_column, max_side=900),
        "gradcam_url": cam_url,
        "gradcam_plus_url": gradcam_plus_url,
    }
