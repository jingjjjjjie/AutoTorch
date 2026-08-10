"""Data-source discovery and schema routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from advanced_visualization.web.models import SchemaResponse, SourceSummary
from advanced_visualization.web.repository import DataSource, repository


router = APIRouter(prefix="/api", tags=["sources"])


def source_summary(source: DataSource, rows: int | None = None) -> SourceSummary:
    return SourceSummary(
        id=source.id,
        label=source.label,
        model_key=source.model_key,
        path=str(source.path),
        rows=rows,
        available=source.path.is_file(),
    )


@router.get("/sources", response_model=list[SourceSummary])
def sources() -> list[SourceSummary]:
    return [
        source_summary(source)
        for source in repository.sources()
        if source.path.is_file()
    ]


@router.get("/sources/{source_id}/schema", response_model=SchemaResponse)
def schema(source_id: str) -> SchemaResponse:
    try:
        details = repository.schema(source_id)
        frame = repository.review_dataframe(source_id)
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return SchemaResponse(
        source=source_summary(details["source"], len(frame)),
        columns=details["columns"],
        numeric_columns=details["numeric_columns"],
        categorical_columns=details["categorical_columns"],
        image_columns=details["image_columns"],
        gradcam_columns=details["gradcam_columns"],
        gradcam_montage_column=details.get("gradcam_montage_column", ""),
        gradcam_montage_layers=details.get("gradcam_montage_layers", []),
        gradcam_montage_layer_labels=details.get(
            "gradcam_montage_layer_labels", {}
        ),
        feature_columns=details["feature_columns"],
        defaults=details["defaults"],
        categories=details["categories"],
        image_availability=details["image_availability"],
        default_filter_columns=details["default_filter_columns"],
        prepared_gradcam_methods=details["prepared_gradcam_methods"],
        prepared_gradcam_layers=details.get("prepared_gradcam_layers", []),
        prepared_gradcam_layer_labels=details.get(
            "prepared_gradcam_layer_labels", {}
        ),
        default_gradcam_layer=details.get("default_gradcam_layer", ""),
        review_preset=details["review_preset"],
    )
