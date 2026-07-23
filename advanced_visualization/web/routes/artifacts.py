"""Validated image and prepared-CAM delivery routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response

from advanced_visualization.web.artifacts import prepared_gradcam_path
from advanced_visualization.web.images import image_bytes
from advanced_visualization.web.repository import repository


router = APIRouter(prefix="/api", tags=["artifacts"])


def jpeg_response(path, max_side: int) -> Response:
    try:
        content, etag = image_bytes(path, max_side=max_side)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return Response(
        content,
        media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=3600", "ETag": etag},
    )


@router.get("/gradcam/{source_id}/{row_id}")
def prepared_gradcam(
    source_id: str,
    row_id: int,
    image_column: str = Query(...),
    method: str = Query("gradcam", pattern=r"^(gradcam|gradcam\+\+)$"),
    max_side: int = Query(900, ge=0, le=4096),
    target: str = "fraud",
    layer: str = "",
) -> Response:
    try:
        source = repository.source(source_id)
        frame = repository.review_dataframe(source_id)
        schema_details = repository.schema(source_id)
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if image_column not in schema_details["image_columns"]:
        raise HTTPException(
            status_code=400, detail="Column is not an image-path column."
        )
    if target not in {"fraud", "genuine"}:
        raise HTTPException(
            status_code=400, detail="CAM target must be fraud or genuine."
        )
    if row_id < 0 or row_id >= len(frame):
        raise HTTPException(status_code=404, detail="Row does not exist.")
    available_layers = set(schema_details.get("prepared_gradcam_layers", []))
    if layer and layer not in available_layers:
        raise HTTPException(status_code=400, detail="Unknown CAM layer for this model.")
    path = prepared_gradcam_path(
        source, frame.iloc[row_id], image_column, method, target, layer
    )
    if path is None:
        raise HTTPException(
            status_code=404,
            detail=f"Prepared {target} {method} does not exist for this row.",
        )
    return jpeg_response(path, max_side)


@router.get("/images/{source_id}/{row_id}")
def image(
    source_id: str,
    row_id: int,
    column: str = Query(...),
    max_side: int = Query(900, ge=0, le=4096),
) -> Response:
    try:
        frame = repository.review_dataframe(source_id)
        schema_details = repository.schema(source_id)
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if column not in set(schema_details["image_columns"]):
        raise HTTPException(
            status_code=400, detail="Column is not an image-path column."
        )
    if (
        row_id < 0
        or row_id >= len(frame)
        or int(frame.iloc[row_id]["__row_id"]) != row_id
    ):
        raise HTTPException(status_code=404, detail="Row does not exist.")
    return jpeg_response(frame.iloc[row_id][column], max_side)
