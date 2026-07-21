"""Feature-projection route."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from advanced_visualization.web.models import ProjectionRequest
from advanced_visualization.web.projections import projection_service
from advanced_visualization.web.repository import repository


router = APIRouter(prefix="/api", tags=["projections"])


@router.post("/projection")
def projection(request: ProjectionRequest) -> dict:
    try:
        source = repository.source(request.source_id)
        frame = repository.dataframe(request.source_id)
        stat = source.path.stat()
        version = f"{stat.st_mtime_ns}:{stat.st_size}"
        return projection_service.project(frame, request, version)
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
