"""Typed HTTP contracts for the visualization API."""
from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field


class FilterRequest(BaseModel):
    source_id: str
    item_id_column: str = ""
    image_column: str = ""
    gradcam_column: str = ""
    gradcam_method: Literal["gradcam", "gradcam++"] = "gradcam"
    gradcam_target: Literal["fraud", "genuine"] = "fraud"
    subclass_column: str = ""
    truth_column: str = ""
    prediction_column: str = ""
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    truth_rows: Literal["valid", "positive", "negative", "invalid", "all"] = "all"
    failure_mode: Literal[
        "all", "failures", "high_confidence", "low_confidence",
        "false_positives", "false_negatives", "correct",
    ] = "all"
    high_confidence: float = Field(default=0.9, ge=0.0, le=1.0)
    low_confidence: float = Field(default=0.6, ge=0.0, le=1.0)
    search: str = ""
    search_columns: list[str] = Field(default_factory=list)
    categorical_filters: dict[str, list[str]] = Field(default_factory=dict)
    sort: Literal[
        "confidence_desc", "confidence_asc", "prediction_desc",
        "prediction_asc", "row_order",
    ] = "confidence_desc"
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=48, ge=12, le=144)


ClassRowLimit = Annotated[int, Field(ge=1, le=50000)]


class ProjectionRequest(BaseModel):
    source_id: str
    method: Literal["pca", "tsne", "umap", "lda"] = "pca"
    feature_columns: list[str] = Field(default_factory=list)
    item_id_column: str = ""
    color_column: str = ""
    categorical_filters: dict[str, list[str]] = Field(default_factory=dict)
    scale: bool = True
    max_rows: int = Field(default=5000, ge=3, le=50000)
    max_rows_per_class: int | None = Field(default=None, ge=1, le=50000)
    max_rows_by_class: dict[str, ClassRowLimit] = Field(default_factory=dict)
    perplexity: int = Field(default=30, ge=2, le=100)
    umap_neighbors: int = Field(default=15, ge=2, le=200)
    umap_min_dist: float = Field(default=0.1, ge=0.0, le=0.99)
    random_state: int = 42


ComparisonOutcome = Literal[
    "both_correct", "a_only_correct", "b_only_correct", "both_wrong",
    "unscored", "truth_mismatch", "only_in_a", "only_in_b",
]


class ComparisonRequest(BaseModel):
    source_a_id: str
    source_b_id: str
    item_id_column_a: str
    item_id_column_b: str
    truth_column_a: str
    truth_column_b: str
    prediction_column_a: str
    prediction_column_b: str
    image_column_a: str = ""
    image_column_b: str = ""
    gradcam_column_a: str = ""
    gradcam_column_b: str = ""
    gradcam_method: Literal["gradcam", "gradcam++"] = "gradcam"
    gradcam_target: Literal["fraud", "genuine"] = "fraud"
    threshold_a: float = Field(default=0.5, ge=0.0, le=1.0)
    threshold_b: float = Field(default=0.5, ge=0.0, le=1.0)
    subclass_column: str = ""
    identity_column: str = ""
    quality_column: str = ""
    categorical_filters: dict[str, list[str]] = Field(default_factory=dict)
    outcomes: list[ComparisonOutcome] = Field(default_factory=list)
    search: str = ""
    sort: Literal[
        "true_confidence_delta_desc", "true_confidence_delta_asc",
        "score_delta_desc", "score_delta_asc", "item_id",
    ] = "true_confidence_delta_desc"
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=24, ge=12, le=144)


AnalysisOutcome = Literal[
    "all", "ori_wrong", "crop_wrong", "merged_wrong", "ori_only_correct",
    "crop_only_correct", "merged_fixes_both", "merged_regression", "ori_crop_disagree",
]


class AnalysisRequest(BaseModel):
    source_id: str
    item_id_column: str
    truth_column: str
    ori_prediction_column: str
    crop_prediction_column: str
    original_image_column: str = ""
    crop_image_column: str = ""
    subclass_column: str = ""
    identity_column: str = ""
    quality_column: str = ""
    quality_mode: Literal["all", "known", "high_quality", "quality_issue"] = "all"
    exclude_unknown_subclass: bool = True
    excluded_identities: list[str] = Field(default_factory=list)
    categorical_filters: dict[str, list[str]] = Field(default_factory=dict)
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    outcome: AnalysisOutcome = "all"
    search: str = ""
    sort: Literal[
        "merged_confidence_desc", "ori_crop_delta_desc", "ori_crop_delta_asc", "item_id",
    ] = "merged_confidence_desc"
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=24, ge=12, le=144)


class SourceSummary(BaseModel):
    id: str
    label: str
    model_key: str
    path: str
    rows: int | None = None
    available: bool = True
    error: str = ""


class SchemaResponse(BaseModel):
    source: SourceSummary
    columns: list[str]
    numeric_columns: list[str]
    categorical_columns: list[str]
    image_columns: list[str]
    gradcam_columns: list[str]
    feature_columns: list[str]
    defaults: dict[str, str]
    categories: dict[str, list[str]]
    image_availability: dict[str, float]
    default_filter_columns: list[str]
    prepared_gradcam_methods: list[str]
    review_preset: dict[str, Any] = Field(default_factory=dict)


JsonValue = str | int | float | bool | None


class RowResponse(BaseModel):
    row_id: int
    values: dict[str, JsonValue]
    image_url: str = ""
    gradcam_url: str = ""


class PageResponse(BaseModel):
    total: int
    page: int
    pages: int
    page_size: int
    metrics: dict[str, Any]
    rows: list[RowResponse]
