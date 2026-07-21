"""Model-comparison and results-analysis routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from advanced_visualization.web.analysis import analysis_page
from advanced_visualization.web.comparisons import comparison_page
from advanced_visualization.web.models import AnalysisRequest, ComparisonRequest
from advanced_visualization.web.repository import repository


router = APIRouter(prefix="/api", tags=["experiments"])


@router.post("/comparison")
def comparison(request: ComparisonRequest) -> dict:
    try:
        return comparison_page(repository, request)
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/analysis")
def analysis(request: AnalysisRequest) -> dict:
    try:
        return analysis_page(repository, request)
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
