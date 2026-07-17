"""FastAPI entrypoint for the non-Streamlit visualization app."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from advanced_visualization.web.filtering import filter_frame, page_frame
from advanced_visualization.web.images import image_bytes
from advanced_visualization.web.models import (
    FilterRequest,
    PageResponse,
    ProjectionRequest,
    SchemaResponse,
    SourceSummary,
)
from advanced_visualization.web.projections import projection_service
from advanced_visualization.web.repository import DataSource, repository


STATIC_DIR = Path(__file__).with_name("static")
app = FastAPI(title="AutoTorch Visualization", version="1.0.0")


def _source_summary(source: DataSource, rows: int | None = None) -> SourceSummary:
    return SourceSummary(
        id=source.id, label=source.label, model_key=source.model_key,
        path=str(source.path), rows=rows, available=source.path.is_file(),
    )


def _json_value(value: Any) -> str | int | float | bool | None:
    if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, int):
        return value
    return str(value)


def _image_url(source_id: str, row_id: int, column: str) -> str:
    return f"/api/images/{source_id}/{row_id}?column={quote(column, safe='')}" if column else ""


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/sources", response_model=list[SourceSummary])
def sources() -> list[SourceSummary]:
    return [_source_summary(source) for source in repository.sources() if source.path.is_file()]


@app.get("/api/sources/{source_id}/schema", response_model=SchemaResponse)
def schema(source_id: str) -> SchemaResponse:
    try:
        details = repository.schema(source_id)
        frame = repository.dataframe(source_id)
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return SchemaResponse(
        source=_source_summary(details["source"], len(frame)),
        columns=details["columns"], numeric_columns=details["numeric_columns"],
        categorical_columns=details["categorical_columns"], image_columns=details["image_columns"],
        gradcam_columns=details["gradcam_columns"], feature_columns=details["feature_columns"],
        defaults=details["defaults"], categories=details["categories"],
    )


@app.post("/api/review", response_model=PageResponse)
def review(request: FilterRequest) -> dict:
    try:
        frame = repository.dataframe(request.source_id)
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    filtered = filter_frame(frame, request)
    page, metadata = page_frame(filtered, request)
    visible_columns = list(dict.fromkeys(filter(None, [
        request.item_id_column, request.subclass_column, request.truth_column,
        request.prediction_column, "__failure_type", "__confidence",
    ])))
    rows = []
    for _, row in page.iterrows():
        row_id = int(row["__row_id"])
        rows.append({
            "row_id": row_id,
            "values": {column: _json_value(row[column]) for column in visible_columns if column in row.index},
            "image_url": _image_url(request.source_id, row_id, request.image_column),
            "gradcam_url": _image_url(request.source_id, row_id, request.gradcam_column),
        })
    return metadata | {"rows": rows}


@app.post("/api/projection")
def projection(request: ProjectionRequest) -> dict:
    try:
        source = repository.source(request.source_id)
        frame = repository.dataframe(request.source_id)
        version = f"{source.path.stat().st_mtime_ns}:{source.path.stat().st_size}"
        result = projection_service.project(frame, request, version)
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if request.image_column in frame.columns:
        for point in result["points"]:
            point["image_url"] = _image_url(request.source_id, point["row_id"], request.image_column)
    return result


@app.get("/api/images/{source_id}/{row_id}")
def image(source_id: str, row_id: int, column: str = Query(...), max_side: int = Query(900, ge=0, le=4096)) -> Response:
    try:
        frame = repository.dataframe(source_id)
        schema_details = repository.schema(source_id)
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    allowed = set(schema_details["image_columns"])
    if column not in allowed:
        raise HTTPException(status_code=400, detail="Column is not an image-path column.")
    match = frame[frame["__row_id"].eq(row_id)]
    if match.empty:
        raise HTTPException(status_code=404, detail="Row does not exist.")
    try:
        content, etag = image_bytes(match.iloc[0][column], max_side=max_side)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return Response(content, media_type="image/jpeg", headers={
        "Cache-Control": "public, max-age=3600", "ETag": etag,
    })


app.mount("/assets", StaticFiles(directory=STATIC_DIR), name="assets")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


def main() -> None:
    import uvicorn

    uvicorn.run("advanced_visualization.web.app:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
