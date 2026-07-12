"""Settings page for prediction CSV and model run definitions."""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from advanced_visualization.core.config import all_model_runs
from advanced_visualization.core.settings import (
    SETTINGS_PATH,
    UserModelConfig,
    UserSettings,
    configured_path,
    load_settings,
    save_settings,
)
from advanced_visualization.ui.styles import inject_css


VIEWER_COLUMNS = [
    "enabled",
    "key",
    "prediction_csv",
    "feature_csv",
    "artifact_dir",
    "model_type",
    "image_column",
    "prediction_column",
]

PREGENERATION_COLUMNS = [
    "key",
    "checkpoint",
    "weights_epoch",
    "model_name",
    "head_type",
    "image_size",
]


def _model_from_editor_row(row: dict) -> UserModelConfig:
    weights_epoch = row.get("weights_epoch")
    return UserModelConfig(
        key=str(row.get("key", "")).strip(),
        prediction_csv=str(row.get("prediction_csv", "")).strip(),
        feature_csv=str(row.get("feature_csv", "")).strip(),
        artifact_dir=str(row.get("artifact_dir", "")).strip(),
        checkpoint=str(row.get("checkpoint", "")).strip(),
        weights_epoch=int(weights_epoch) if pd.notna(weights_epoch) and str(weights_epoch).strip() else None,
        model_type=str(row.get("model_type", "")).strip(),
        model_name=str(row.get("model_name", "")).strip(),
        head_type=str(row.get("head_type", "")).strip(),
        image_size=int(row.get("image_size") or 0),
        image_column=str(row.get("image_column", "")).strip(),
        prediction_column=str(row.get("prediction_column", "")).strip(),
        enabled=bool(row.get("enabled", True)),
    )


def _settings_dataframe(settings: UserSettings) -> pd.DataFrame:
    rows = [model.to_dict() for model in settings.models]
    if not rows:
        rows = [UserModelConfig().to_dict()]
    return pd.DataFrame(rows)


def _empty_frame(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame([{column: UserModelConfig().to_dict().get(column, "") for column in columns}])


def _render_model_editor(settings: UserSettings) -> list[UserModelConfig]:
    st.subheader("Viewer Data Sources")
    st.caption("These fields are enough for the Streamlit viewer. The viewer reads CSVs and prepared images; it does not load model weights.")
    source_df = _settings_dataframe(settings)
    source_edited = st.data_editor(
        source_df[VIEWER_COLUMNS] if not source_df.empty else _empty_frame(VIEWER_COLUMNS),
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config={
            "enabled": st.column_config.CheckboxColumn("Enabled"),
            "key": st.column_config.TextColumn("Model key", required=True),
            "prediction_csv": st.column_config.TextColumn("Prediction CSV"),
            "feature_csv": st.column_config.TextColumn("Feature CSV"),
            "artifact_dir": st.column_config.TextColumn("Artifact directory"),
            "model_type": st.column_config.SelectboxColumn("Model type", options=settings.model_type_options),
            "image_column": st.column_config.TextColumn("Image column"),
            "prediction_column": st.column_config.TextColumn("Prediction column"),
        },
        key="advanced_visualization_settings_model_sources",
    )

    st.subheader("Pregeneration / Model Loading")
    st.caption("Only CLI preparation uses these fields. Keep them blank for artifact-only models that were generated elsewhere.")
    generation_df = _settings_dataframe(settings)
    generation_edited = st.data_editor(
        generation_df[PREGENERATION_COLUMNS] if not generation_df.empty else _empty_frame(PREGENERATION_COLUMNS),
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config={
            "key": st.column_config.TextColumn("Model key", required=True),
            "checkpoint": st.column_config.TextColumn("Checkpoint path"),
            "weights_epoch": st.column_config.NumberColumn("Epoch", min_value=0, step=1),
            "model_name": st.column_config.TextColumn("Model name"),
            "head_type": st.column_config.TextColumn("Head type"),
            "image_size": st.column_config.SelectboxColumn("Image size", options=settings.image_size_options),
        },
        key="advanced_visualization_settings_model_generation",
    )

    generation_by_key = {
        str(row.get("key", "")).strip(): row
        for row in generation_edited.to_dict("records")
        if str(row.get("key", "")).strip()
    }
    models = []
    source_keys = set()
    for source_row in source_edited.to_dict("records"):
        key = str(source_row.get("key", "")).strip()
        source_keys.add(key)
        row = {**source_row, **generation_by_key.get(key, {})}
        model = _model_from_editor_row(row)
        if model.key or model.artifact_dir or model.checkpoint or model.prediction_column:
            models.append(model)

    for key, generation_row in generation_by_key.items():
        if key in source_keys:
            continue
        model = _model_from_editor_row(generation_row)
        if model.key or model.checkpoint:
            models.append(model)
    return models


def _render_review_settings(settings: UserSettings) -> dict:
    current = settings.review
    st.subheader("Image Review Defaults")
    left, middle, right = st.columns(3)
    with left:
        positive_value = st.number_input("Positive truth value", value=int(current.get("positive_truth_value", 1)), step=1)
        negative_value = st.number_input("Negative truth value", value=int(current.get("negative_truth_value", 0)), step=1)
        invalid_value = st.number_input("Invalid truth value", value=int(current.get("invalid_truth_value", -1)), step=1)
    with middle:
        prediction_threshold = st.slider(
            "Prediction threshold",
            min_value=0.0,
            max_value=1.0,
            value=float(current.get("prediction_threshold", 0.5)),
            step=0.01,
        )
        high_confidence = st.slider(
            "High confidence",
            min_value=0.0,
            max_value=1.0,
            value=float(current.get("high_confidence", 0.9)),
            step=0.01,
        )
        low_confidence = st.slider(
            "Low confidence",
            min_value=0.0,
            max_value=1.0,
            value=float(current.get("low_confidence", 0.6)),
            step=0.01,
        )
    with right:
        truth_row_options = ["Valid only: 0 and 1", "Positive only: 1", "Negative only: 0", "Invalid only: -1", "All rows"]
        failure_view_options = [
            "All rows",
            "Failures only",
            "High-confidence failures",
            "Low-confidence failures",
            "False positives",
            "False negatives",
            "Correct only",
        ]
        default_truth_rows = st.selectbox(
            "Default truth rows",
            truth_row_options,
            index=truth_row_options.index(current.get("default_truth_rows", "Valid only: 0 and 1"))
            if current.get("default_truth_rows", "Valid only: 0 and 1") in truth_row_options
            else 0,
        )
        default_failure_view = st.selectbox(
            "Default failure view",
            failure_view_options,
            index=failure_view_options.index(current.get("default_failure_view", "All rows"))
            if current.get("default_failure_view", "All rows") in failure_view_options
            else 0,
        )
        only_prepared_gradcam = st.checkbox("Only prepared Grad-CAM by default", value=bool(current.get("only_prepared_gradcam", False)))

    st.subheader("Image Review Layout")
    layout_cols = st.columns(4)
    with layout_cols[0]:
        browse_mode_options = ["Bottomless scroll", "Pages"]
        default_browse_mode = st.selectbox(
            "Browse mode",
            browse_mode_options,
            index=browse_mode_options.index(current.get("default_browse_mode", "Bottomless scroll"))
            if current.get("default_browse_mode", "Bottomless scroll") in browse_mode_options
            else 0,
        )
    with layout_cols[1]:
        image_mode_options = ["Original", "Grad-CAM", "Side-by-side"]
        default_image_mode = st.selectbox(
            "Image mode",
            image_mode_options,
            index=image_mode_options.index(current.get("default_image_mode", "Original"))
            if current.get("default_image_mode", "Original") in image_mode_options
            else 0,
        )
    with layout_cols[2]:
        page_size = st.selectbox("Page size", [12, 24, 48, 96, 144], index=[12, 24, 48, 96, 144].index(int(current.get("page_size", 48))))
    with layout_cols[3]:
        columns_per_row = st.slider("Columns per row", min_value=2, max_value=10, value=int(current.get("columns_per_row", 6)))

    sort_options = ["confidence desc", "confidence asc", "prediction desc", "prediction asc", "row order"]
    default_sort = st.selectbox(
        "Default sort",
        sort_options,
        index=sort_options.index(current.get("default_sort", "confidence desc"))
        if current.get("default_sort", "confidence desc") in sort_options
        else 0,
    )
    default_filter_columns = st.text_input(
        "Default categorical filter columns",
        value=",".join(current.get("default_filter_columns", ["Recapture_Subclass", "Data_Identity", "Quality_Issue"])),
        help="Comma-separated column names. Missing columns are ignored.",
    )

    return {
        "positive_truth_value": int(positive_value),
        "negative_truth_value": int(negative_value),
        "invalid_truth_value": int(invalid_value),
        "default_truth_rows": default_truth_rows,
        "prediction_threshold": float(prediction_threshold),
        "high_confidence": float(high_confidence),
        "low_confidence": float(low_confidence),
        "default_failure_view": default_failure_view,
        "default_browse_mode": default_browse_mode,
        "default_image_mode": default_image_mode,
        "only_prepared_gradcam": bool(only_prepared_gradcam),
        "page_size": int(page_size),
        "columns_per_row": int(columns_per_row),
        "default_sort": default_sort,
        "default_filter_columns": [value.strip() for value in default_filter_columns.split(",") if value.strip()],
    }


def _render_extra_view_configs(settings: UserSettings) -> list[dict]:
    st.subheader("Workspace Configs")
    st.caption("Defines model-specific launchable workspaces, branches, layers, and Grad-CAM path column templates.")
    raw = st.text_area(
        "Workspace configs JSON",
        value=json.dumps(settings.extra_view_configs, indent=2),
        height=360,
    )
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        st.warning(f"Workspace configs JSON is invalid: {exc}")
        return settings.extra_view_configs
    if not isinstance(parsed, list):
        st.warning("Workspace configs must be a JSON list.")
        return settings.extra_view_configs
    return [item for item in parsed if isinstance(item, dict)]


def _validate(settings: UserSettings) -> list[str]:
    errors: list[str] = []
    if settings.prediction_csv and not configured_path(settings.prediction_csv).is_file():
        errors.append(f"Prediction CSV does not exist: {settings.prediction_csv}")

    keys = set()
    for index, model in enumerate(settings.models, start=1):
        label = model.key or f"row {index}"
        if not model.enabled:
            continue
        prediction_csv = model.prediction_csv or settings.prediction_csv
        if not prediction_csv:
            errors.append(f"{label}: prediction_csv is required on the model or globally.")
        elif not configured_path(prediction_csv).is_file():
            errors.append(f"{label}: prediction CSV does not exist: {prediction_csv}")
        if model.feature_csv and not configured_path(model.feature_csv).is_file():
            errors.append(f"{label}: feature CSV does not exist: {model.feature_csv}")
        if not model.key:
            errors.append(f"Model {index}: key is required.")
        elif model.key in keys:
            errors.append(f"Duplicate model key: {model.key}")
        keys.add(model.key)
        if not model.artifact_dir:
            errors.append(f"{label}: artifact_dir is required.")
        if not model.prediction_column:
            errors.append(f"{label}: prediction_column is required.")
        if model.model_type and model.model_type != "artifact_only" and model.resolved_checkpoint() is None:
            errors.append(f"{label}: provide checkpoint or artifact_dir + epoch before running pregeneration.")
    return errors


def main() -> None:
    inject_css()
    st.markdown(
        """
        <div class="app-hero">
          <h1>Settings</h1>
          <p>Define viewer data sources separately from CLI-only model loading fields.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    current = load_settings()
    prediction_csv = st.text_input("Prediction CSV", value=current.prediction_csv)
    review = _render_review_settings(current)
    models = _render_model_editor(current)
    extra_view_configs = _render_extra_view_configs(current)
    next_settings = UserSettings(
        prediction_csv=prediction_csv.strip(),
        manifest_name=current.manifest_name,
        prepared_csv_name=current.prepared_csv_name,
        default_gradcam_root=current.default_gradcam_root,
        image_columns=current.image_columns,
        id_columns=current.id_columns,
        subclass_columns=current.subclass_columns,
        image_extensions=current.image_extensions,
        normalize_mean=current.normalize_mean,
        normalize_std=current.normalize_std,
        review=review,
        model_type_options=current.model_type_options,
        image_size_options=current.image_size_options,
        extra_view_configs=extra_view_configs,
        pipeline=current.pipeline,
        models=models,
    )

    errors = _validate(next_settings)
    if errors:
        for error in errors:
            st.warning(error)

    left, right = st.columns([0.25, 0.75])
    with left:
        if st.button("Save settings", type="primary", use_container_width=True):
            save_settings(next_settings)
            st.success(f"Saved to {SETTINGS_PATH}")
            st.rerun()
    with right:
        st.caption(f"Settings file: {SETTINGS_PATH}")
        st.caption("Prepare feature CSVs and Grad-CAM files outside the Streamlit viewer.")

    with st.expander("Active model configs", expanded=False):
        active = [config.to_json_dict() for config in all_model_runs().values()]
        st.dataframe(pd.DataFrame(active), hide_index=True, use_container_width=True)
