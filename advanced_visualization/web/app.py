"""FastAPI entrypoint for the non-Streamlit visualization app."""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from advanced_visualization.web.analysis import analysis_page
from advanced_visualization.web.artifacts import (
    image_url as _image_url,
    prepared_gradcam_path as _prepared_gradcam_path,
    prepared_gradcam_url as _prepared_gradcam_url,
)
from advanced_visualization.web.comparisons import comparison_page
from advanced_visualization.web.filtering import filter_frame, page_frame
from advanced_visualization.web.images import image_bytes
from advanced_visualization.web.models import (
    AnalysisRequest,
    ComparisonRequest,
    FilterRequest,
    PageResponse,
    ProjectionRequest,
    SchemaResponse,
    SourceSummary,
)
from advanced_visualization.web.projections import projection_service
from advanced_visualization.web.repository import DataSource, repository
from advanced_visualization.web.serialization import json_value as _json_value


STATIC_DIR = Path(__file__).with_name("static")
app = FastAPI(title="AutoTorch Visualization", version="1.0.0")


def _source_summary(source: DataSource, rows: int | None = None) -> SourceSummary:
    return SourceSummary(
        id=source.id, label=source.label, model_key=source.model_key,
        path=str(source.path), rows=rows, available=source.path.is_file(),
    )


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
        frame = repository.review_dataframe(source_id)
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return SchemaResponse(
        source=_source_summary(details["source"], len(frame)),
        columns=details["columns"], numeric_columns=details["numeric_columns"],
        categorical_columns=details["categorical_columns"], image_columns=details["image_columns"],
        gradcam_columns=details["gradcam_columns"], feature_columns=details["feature_columns"],
        defaults=details["defaults"], categories=details["categories"],
        image_availability=details["image_availability"],
        default_filter_columns=details["default_filter_columns"],
        prepared_gradcam_methods=details["prepared_gradcam_methods"],
        review_preset=details["review_preset"],
    )


@app.post("/api/review", response_model=PageResponse)
def review(request: FilterRequest) -> dict:
    try:
        source = repository.source(request.source_id)
        frame = repository.review_dataframe(request.source_id)
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    requested_columns = {
        "__row_id", request.item_id_column, request.subclass_column,
        request.truth_column, request.prediction_column, request.image_column,
        request.gradcam_column,
        *request.search_columns, *request.categorical_filters.keys(),
    }
    requested_columns.discard("")
    review_frame = frame.loc[:, [column for column in frame.columns if column in requested_columns]]
    filtered = filter_frame(review_frame, request)
    page, metadata = page_frame(filtered, request)
    visible_columns = list(dict.fromkeys(filter(None, [
        request.item_id_column, request.subclass_column, request.truth_column,
        request.prediction_column, "__failure_type", "__confidence",
    ])))
    rows = []
    for _, row in page.iterrows():
        row_id = int(row["__row_id"])
        gradcam_url = (
            _image_url(request.source_id, row_id, request.gradcam_column)
            if request.gradcam_target == "fraud" else ""
        )
        if not gradcam_url and request.image_column:
            path = _prepared_gradcam_path(
                source, row, request.image_column, request.gradcam_method, request.gradcam_target
            )
            if path is not None:
                gradcam_url = _prepared_gradcam_url(
                    request.source_id, row_id, request.image_column,
                    request.gradcam_method, request.gradcam_target,
                )
        rows.append({
            "row_id": row_id,
            "values": {column: _json_value(row[column]) for column in visible_columns if column in row.index},
            "image_url": _image_url(request.source_id, row_id, request.image_column),
            "gradcam_url": gradcam_url,
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
    return result


@app.post("/api/comparison")
def comparison(request: ComparisonRequest) -> dict:
    try:
        return comparison_page(repository, request)
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/analysis")
def analysis(request: AnalysisRequest) -> dict:
    try:
        return analysis_page(repository, request)
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/api/points/{source_id}/{row_id}")
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
        raise HTTPException(status_code=400, detail="CAM target must be fraud or genuine.")
    row = frame.iloc[row_id]
    allowed_images = set(schema_details["image_columns"])
    image_column = image_column if image_column in allowed_images else ""
    gradcam_column = gradcam_column if gradcam_column in allowed_images else ""
    values = {}
    for column in (item_id_column, group_column, prediction_column):
        if column and column in row.index:
            values[column] = _json_value(row[column])

    gradcam_url = (
        _image_url(source_id, row_id, gradcam_column, max_side=900)
        if gradcam_column and gradcam_target == "fraud" else ""
    )
    gradcam_plus_url = ""
    if image_column:
        if not gradcam_url and _prepared_gradcam_path(
            source, row, image_column, "gradcam", gradcam_target
        ):
            gradcam_url = _prepared_gradcam_url(
                source_id, row_id, image_column, "gradcam", gradcam_target
            )
        if _prepared_gradcam_path(source, row, image_column, "gradcam++", gradcam_target):
            gradcam_plus_url = _prepared_gradcam_url(
                source_id, row_id, image_column, "gradcam++", gradcam_target
            )
    return {
        "row_id": row_id,
        "values": values,
        "image_url": _image_url(source_id, row_id, image_column, max_side=900),
        "gradcam_url": gradcam_url,
        "gradcam_plus_url": gradcam_plus_url,
    }


@app.get("/api/gradcam/{source_id}/{row_id}")
def prepared_gradcam(
    source_id: str,
    row_id: int,
    image_column: str = Query(...),
    method: str = Query("gradcam", pattern=r"^(gradcam|gradcam\+\+)$"),
    max_side: int = Query(900, ge=0, le=4096),
    target: str = "fraud",
) -> Response:
    try:
        source = repository.source(source_id)
        frame = repository.review_dataframe(source_id)
        schema_details = repository.schema(source_id)
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if image_column not in schema_details["image_columns"]:
        raise HTTPException(status_code=400, detail="Column is not an image-path column.")
    if target not in {"fraud", "genuine"}:
        raise HTTPException(status_code=400, detail="CAM target must be fraud or genuine.")
    if row_id < 0 or row_id >= len(frame):
        raise HTTPException(status_code=404, detail="Row does not exist.")
    path = _prepared_gradcam_path(source, frame.iloc[row_id], image_column, method, target)
    if path is None:
        raise HTTPException(
            status_code=404, detail=f"Prepared {target} {method} does not exist for this row."
        )
    try:
        content, etag = image_bytes(path, max_side=max_side)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return Response(content, media_type="image/jpeg", headers={
        "Cache-Control": "public, max-age=3600", "ETag": etag,
    })


@app.get("/api/images/{source_id}/{row_id}")
def image(source_id: str, row_id: int, column: str = Query(...), max_side: int = Query(900, ge=0, le=4096)) -> Response:
    try:
        frame = repository.review_dataframe(source_id)
        schema_details = repository.schema(source_id)
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    allowed = set(schema_details["image_columns"])
    if column not in allowed:
        raise HTTPException(status_code=400, detail="Column is not an image-path column.")
    if row_id < 0 or row_id >= len(frame) or int(frame.iloc[row_id]["__row_id"]) != row_id:
        raise HTTPException(status_code=404, detail="Row does not exist.")
    try:
        content, etag = image_bytes(frame.iloc[row_id][column], max_side=max_side)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return Response(content, media_type="image/jpeg", headers={
        "Cache-Control": "public, max-age=3600", "ETag": etag,
    })


app.mount("/assets", StaticFiles(directory=STATIC_DIR), name="assets")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    return Response(status_code=204)


def main() -> None:
    import uvicorn

    uvicorn.run("advanced_visualization.web.app:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
