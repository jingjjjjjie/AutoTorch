"""Sidebar controls for the image-review view."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from advanced_visualization.core.artifacts import load_manifest
from advanced_visualization.core.columns import (
    GRADCAM_PATTERN,
    categorical_columns,
    first_existing,
    image_path_columns,
    numeric_filter_columns,
    prediction_columns,
)
from advanced_visualization.core.config import (
    ID_COLUMNS,
    IMAGE_COLUMNS,
    SUBCLASS_COLUMNS,
    all_model_runs,
    gradcam_artifact_root,
)
from advanced_visualization.core.settings import load_settings


def option_index(options: list, value, fallback: int = 0) -> int:
    return options.index(value) if value in options else fallback


def configured_prediction_column(model_key: str, columns: list[str]) -> str | None:
    if not model_key:
        return None
    active_model = all_model_runs().get(model_key)
    if active_model and active_model.prediction_column in columns:
        return active_model.prediction_column
    for model in load_settings().models:
        if model.key == model_key and model.prediction_column in columns:
            return model.prediction_column
    return None


def sidebar_controls(df: pd.DataFrame) -> dict:
    settings = load_settings()
    review = settings.review
    columns = df.columns.astype(str).tolist()
    image_columns = image_path_columns(df)
    pred_columns = prediction_columns(df)
    cats = categorical_columns(df)
    artifact_dir = st.session_state.get("advanced_visualization_artifact_dir")
    active_csv_stem = st.session_state.get("advanced_visualization_active_csv_stem")
    manifest = load_manifest(Path(artifact_dir)) if artifact_dir else None

    id_default = manifest.item_id_column if manifest and manifest.item_id_column in columns else first_existing(columns, ID_COLUMNS) or columns[0]
    image_default = manifest.image_column if manifest and manifest.image_column in image_columns else first_existing(columns, IMAGE_COLUMNS)
    subclass_default = manifest.subclass_column if manifest and manifest.subclass_column in cats else first_existing(columns, SUBCLASS_COLUMNS)
    truth_default = manifest.truth_column if manifest and manifest.truth_column in columns else ("label" if "label" in df.columns else subclass_default)
    configured_pred_default = configured_prediction_column(str(active_csv_stem), pred_columns) if active_csv_stem else None
    pred_default = manifest.prediction_column if manifest and manifest.prediction_column in pred_columns else (configured_pred_default or (pred_columns[-1] if pred_columns else None))
    gradcam_candidates = [column for column in image_columns if GRADCAM_PATTERN.search(str(column))]

    default_gradcam_dir = manifest.gradcam_dir if manifest else (gradcam_artifact_root(active_csv_stem) if active_csv_stem else None)

    with st.sidebar.form("advanced_visualization_filter_form"):
        submitted = st.form_submit_button("Update filters", type="primary", use_container_width=True)

        st.header("Columns")
        item_id_column = st.selectbox("Item ID", columns, index=columns.index(id_default))
        image_column = st.selectbox(
            "Image path",
            ["None"] + image_columns,
            index=(image_columns.index(image_default) + 1 if image_default in image_columns else 0),
        )
        subclass_column = st.selectbox(
            "Subclass/group",
            ["None"] + cats,
            index=(cats.index(subclass_default) + 1 if subclass_default in cats else 0),
        )
        truth_column = st.selectbox(
            "Truth column",
            ["None"] + columns,
            index=(columns.index(truth_default) + 1 if truth_default in columns else 0),
        )
        prediction_column = st.selectbox(
            "Prediction score",
            ["None"] + pred_columns,
            index=(pred_columns.index(pred_default) + 1 if pred_default in pred_columns else 0),
        )
        gradcam_column = st.selectbox(
            "Grad-CAM path column",
            ["None"] + image_columns,
            index=(image_columns.index(gradcam_candidates[0]) + 1 if gradcam_candidates else 0),
        )
        gradcam_dir = st.text_input(
            "Grad-CAM directory",
            value=str(default_gradcam_dir) if default_gradcam_dir and default_gradcam_dir.exists() else "",
            help="Optional fallback: looks for image-stem PNG/JPG files in this directory.",
            key=f"gradcam_dir_{active_csv_stem or 'uploaded'}",
        )

        st.header("Failure Logic")
        prediction_threshold = st.slider("Prediction threshold", 0.0, 1.0, float(review.get("prediction_threshold", 0.5)), 0.01)
        truth_row_options = ["Valid only: 0 and 1", "Positive only: 1", "Negative only: 0", "Invalid only: -1", "All rows"]
        truth_rows = st.radio(
            "Truth rows",
            truth_row_options,
            index=option_index(truth_row_options, review.get("default_truth_rows", "Valid only: 0 and 1")),
        )
        high_conf = st.slider("High confidence >=", 0.0, 1.0, float(review.get("high_confidence", 0.9)), 0.01)
        low_conf = st.slider("Low confidence <=", 0.0, 1.0, float(review.get("low_confidence", 0.6)), 0.01)
        failure_view_options = [
            "All rows",
            "Failures only",
            "High-confidence failures",
            "Low-confidence failures",
            "False positives",
            "False negatives",
            "Correct only",
        ]
        mode = st.radio(
            "Failure view",
            failure_view_options,
            index=option_index(failure_view_options, review.get("default_failure_view", "All rows")),
        )

        st.header("Advanced Filters")
        search_columns = st.multiselect(
            "Search columns",
            columns,
            default=[column for column in [item_id_column, subclass_column] if column and column != "None"],
        )
        text_query = st.text_input("Search text")
        configured_filter_columns = review.get("default_filter_columns", ["Recapture_Subclass", "Data_Identity", "Quality_Issue"])
        default_filter_columns = [
            column
            for column in configured_filter_columns
            if column and column != "None" and column in cats
        ]
        filter_columns = st.multiselect("Categorical filters", cats, default=default_filter_columns)

        categorical_filters = {}
        for column in filter_columns:
            values = sorted(df[column].fillna("Missing").astype(str).unique().tolist())
            categorical_filters[column] = st.multiselect(column, values, default=values)

        numeric_ranges = {}
        with st.expander("Numeric ranges", expanded=False):
            range_columns = st.multiselect("Range columns", numeric_filter_columns(df), default=[])
            for column in range_columns:
                values = pd.to_numeric(df[column], errors="coerce").dropna()
                if values.empty:
                    continue
                min_value = float(values.min())
                max_value = float(values.max())
                if min_value == max_value:
                    st.caption(f"{column}: constant {min_value:g}")
                    numeric_ranges[column] = (min_value, max_value)
                    continue
                numeric_ranges[column] = st.slider(column, min_value, max_value, (min_value, max_value))

        st.header("Paging")
        browse_mode_options = ["Bottomless scroll", "Pages"]
        browse_mode = st.radio(
            "Browse mode",
            browse_mode_options,
            index=option_index(browse_mode_options, review.get("default_browse_mode", "Bottomless scroll")),
            horizontal=True,
        )
        image_mode_options = ["Original", "Grad-CAM", "Side-by-side"]
        view_mode = st.radio(
            "Image mode",
            image_mode_options,
            index=option_index(image_mode_options, review.get("default_image_mode", "Original")),
            horizontal=True,
        )
        cam_method = st.radio("CAM method", ["gradcam", "gradcam++"], index=0, horizontal=True)
        cam_space = st.radio(
            "CAM overlay",
            ["original", "model-input"],
            index=0,
            horizontal=True,
            help="original maps CAM back to the source image. model-input shows CAM on the exact transformed image sent to the model.",
        )
        only_prepared_gradcam = st.checkbox(
            "Only prepared Grad-CAM",
            value=bool(review.get("only_prepared_gradcam", False)),
            help="Shows only rows with existing prepared Grad-CAM image files.",
            key=f"only_prepared_gradcam_{view_mode}",
        )
        page_size_options = [12, 24, 48, 96, 144]
        page_size = st.select_slider("Page size", options=page_size_options, value=int(review.get("page_size", 48)))
        columns_per_row = st.slider("Columns per row", 2, 10, int(review.get("columns_per_row", 6)))
        show_card_metadata = st.checkbox("Show card metadata", value=bool(review.get("show_card_metadata", False)))
        sort_options = ["confidence desc", "confidence asc", "prediction desc", "prediction asc", "row order"]
        sort_by = st.selectbox(
            "Sort",
            sort_options,
            index=option_index(sort_options, review.get("default_sort", "confidence desc")),
        )

    if submitted:
        st.session_state["advanced_visualization_visible_count"] = page_size
        st.session_state["advanced_visualization_page"] = 1

    return {
        "item_id_column": item_id_column,
        "image_column": None if image_column == "None" else image_column,
        "subclass_column": None if subclass_column == "None" else subclass_column,
        "truth_column": None if truth_column == "None" else truth_column,
        "prediction_column": None if prediction_column == "None" else prediction_column,
        "gradcam_column": None if gradcam_column == "None" else gradcam_column,
        "gradcam_dir": gradcam_dir,
        "active_csv_stem": active_csv_stem,
        "prediction_threshold": prediction_threshold,
        "truth_rows": truth_rows,
        "positive_truth_value": int(review.get("positive_truth_value", 1)),
        "negative_truth_value": int(review.get("negative_truth_value", 0)),
        "invalid_truth_value": int(review.get("invalid_truth_value", -1)),
        "high_conf": high_conf,
        "low_conf": low_conf,
        "mode": mode,
        "search_columns": search_columns,
        "text_query": text_query,
        "categorical_filters": categorical_filters,
        "numeric_ranges": numeric_ranges,
        "browse_mode": browse_mode,
        "view_mode": view_mode,
        "cam_method": cam_method,
        "cam_space": cam_space,
        "only_prepared_gradcam": only_prepared_gradcam,
        "page_size": page_size,
        "columns_per_row": columns_per_row,
        "show_card_metadata": show_card_metadata,
        "sort_by": sort_by,
    }
