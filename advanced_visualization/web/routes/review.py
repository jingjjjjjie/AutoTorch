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


def target_gradcam_column(
    requested: str, target: str, allowed_images: set[str]
) -> str:
    """Resolve fraud/genuine counterparts for explicitly prepared CAM columns."""
    if not requested or requested not in allowed_images:
        return ""
    paired = requested
    if target == "genuine":
        paired = requested.replace("_fraud_", "_genuine_")
    elif target == "fraud":
        paired = requested.replace("_genuine_", "_fraud_")
    return paired if paired in allowed_images else requested


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
    schema_details = repository.schema(request.source_id)
    allowed_images = set(schema_details["image_columns"])
    available_layers = schema_details.get("prepared_gradcam_layers", [])
    selected_layer = request.gradcam_layer or schema_details.get("default_gradcam_layer", "")
    if selected_layer and selected_layer not in available_layers:
        raise HTTPException(status_code=400, detail="Unknown CAM layer for this model.")

    def resolved_cam(row, row_id: int, target: str, layer: str) -> str:
        selected_column = target_gradcam_column(
            request.gradcam_column, target, allowed_images
        )
        url = image_url(request.source_id, row_id, selected_column)
        if not url and request.image_column and prepared_gradcam_path(
            source,
            row,
            request.image_column,
            request.gradcam_method,
            target,
            layer,
        ):
            url = prepared_gradcam_url(
                request.source_id,
                row_id,
                request.image_column,
                request.gradcam_method,
                target,
                layer,
            )
        return url
    for _, row in page.iterrows():
        row_id = int(row["__row_id"])
        genuine_url = resolved_cam(row, row_id, "genuine", selected_layer)
        fraud_url = resolved_cam(row, row_id, "fraud", selected_layer)
        cam_url = genuine_url if request.gradcam_target == "genuine" else fraud_url
        layer_urls = [
            {
                "layer": layer,
                "genuine_url": resolved_cam(row, row_id, "genuine", layer),
                "fraud_url": resolved_cam(row, row_id, "fraud", layer),
            }
            for layer in available_layers
        ]
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
                "genuine_gradcam_url": genuine_url,
                "fraud_gradcam_url": fraud_url,
                "gradcam_layers": layer_urls,
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
    gradcam_layer: str = "",
) -> dict:
    try:
        source = repository.source(source_id)
        frame = repository.review_dataframe(source_id)
        schema_details = repository.schema(source_id)
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if row_id < 0 or row_id >= len(frame):
        raise HTTPException(status_code=404, detail="Row does not exist.")
    if gradcam_target not in {"fraud", "genuine", "both"}:
        raise HTTPException(
            status_code=400, detail="CAM target must be fraud or genuine."
        )
    row = frame.iloc[row_id]
    allowed_images = set(schema_details["image_columns"])
    image_column = image_column if image_column in allowed_images else ""
    gradcam_column = target_gradcam_column(
        gradcam_column, gradcam_target, allowed_images
    )
    values = {
        column: json_value(row[column])
        for column in (item_id_column, group_column, prediction_column)
        if column and column in row.index
    }

    selected_layer = gradcam_layer or schema_details.get("default_gradcam_layer", "")
    genuine_url = fraud_url = ""
    if image_column:
        for target in ("genuine", "fraud"):
            if prepared_gradcam_path(
                source, row, image_column, "gradcam++", target, selected_layer
            ):
                url = prepared_gradcam_url(
                    source_id, row_id, image_column, "gradcam++", target, selected_layer
                )
                if target == "genuine":
                    genuine_url = url
                else:
                    fraud_url = url
    selected_url = genuine_url if gradcam_target == "genuine" else fraud_url
    return {
        "row_id": row_id,
        "values": values,
        "image_url": image_url(source_id, row_id, image_column, max_side=900),
        "gradcam_url": selected_url,
        "gradcam_plus_url": selected_url,
        "genuine_gradcam_url": genuine_url,
        "fraud_gradcam_url": fraud_url,
    }
