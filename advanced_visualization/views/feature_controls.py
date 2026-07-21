"""Sidebar controls for the legacy Streamlit feature explorer.

The modern FastAPI viewer is the primary interface. Keeping these controls in a
separate module prevents UI state management from obscuring projection and
rendering behavior in the compatibility view.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

from advanced_visualization.core.feature_data import (
    PRED_BUCKET_COLUMN,
    PREDICTION_PATTERN,
    feature_columns,
    metadata_columns,
)
from advanced_visualization.core.settings import load_settings

DEFAULT_METADATA_COLUMNS = ("Recapture_Subclass", "Data_Identity", "Quality_Issue")


def normalize_selection_key(
    key: str, values: list[str], default: Optional[list[str]] = None
) -> list[str]:
    if default is None:
        default = values
    selected = st.session_state.get(key)
    if selected is None:
        selected = default
    selected = [value for value in selected if value in values]
    if not selected and default:
        selected = [value for value in default if value in values]
    st.session_state[key] = selected
    return selected


def find_default_column(
    columns: list[str], candidates: tuple[str, ...]
) -> Optional[str]:
    lowered = {str(column).lower(): column for column in columns}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    return columns[0] if columns else None


def active_prediction_column(df: pd.DataFrame) -> Optional[str]:
    model_key = str(st.session_state.get("active_feature_model_key") or "")
    if model_key:
        for model in load_settings().models:
            if model.key == model_key and model.prediction_column in df.columns:
                return model.prediction_column
    if "__prediction_column" in df.columns:
        return "__prediction_column"

    candidates = [
        column
        for column in df.select_dtypes(include=[np.number]).columns
        if PREDICTION_PATTERN.search(str(column)) and column not in feature_columns(df)
    ]
    if len(candidates) == 1:
        return candidates[0]
    generated_candidates = [
        column for column in candidates if str(column).endswith("_pred")
    ]
    return generated_candidates[-1] if generated_candidates else None


def configured_image_column_for_model(
    model_key: str, columns: list[str]
) -> Optional[str]:
    if not model_key:
        return None
    for model in load_settings().models:
        if model.key == model_key and model.image_column in columns:
            return model.image_column
    return None


def sidebar_controls(df: pd.DataFrame) -> dict:
    st.sidebar.header("Data")
    selected_features = feature_columns(df)
    st.sidebar.caption(
        f"Using {len(selected_features):,} auto-detected feature columns."
    )
    prediction_column = active_prediction_column(df)
    if prediction_column:
        st.sidebar.caption(f"Using model prediction: {prediction_column}")

    available_metadata = [
        column for column in df.columns if column not in selected_features
    ]
    default_metadata = [
        column for column in DEFAULT_METADATA_COLUMNS if column in available_metadata
    ]
    if not default_metadata:
        default_metadata = [
            find_default_column(
                available_metadata, ("class", "sample_type", "source", "label", "batch")
            )
        ]
        default_metadata = [column for column in default_metadata if column is not None]
    selected_metadata_columns = st.sidebar.multiselect(
        "Metadata columns",
        options=available_metadata,
        default=default_metadata,
        help="Keep this small for large CSVs. These columns become group/filter/color options.",
    )

    cats = metadata_columns(df, selected_features, selected_metadata_columns)
    if not cats:
        raise ValueError(
            "At least one categorical metadata column is required for grouping."
        )

    id_candidates = ("id", "uuid")
    id_default_column = find_default_column(df.columns.tolist(), id_candidates)
    id_default = (
        df.columns.tolist().index(id_default_column)
        if id_default_column in df.columns
        else 0
    )
    item_id_column = st.sidebar.selectbox(
        "Item identifier", options=df.columns.tolist(), index=id_default
    )

    default_group = find_default_column(
        cats, ("recapture_subclass", "class", "sample_type", "source", "label", "batch")
    )
    group_column = st.sidebar.selectbox(
        "Primary class/group",
        options=cats,
        index=cats.index(default_group) if default_group in cats else 0,
    )

    merge_rules = st.sidebar.text_area(
        "Merge classes",
        placeholder="colour printed=collected colour printed,printed colour prod\nmix and match=mix,match",
        help="One rule per line. Values not listed are kept unchanged.",
    )

    filters = {}
    filter_specs = []
    for column in cats:
        values = sorted(df[column].dropna().astype(str).unique().tolist())
        if not values or len(values) > 120:
            continue
        applied_key = f"filter_applied_{column}"
        pending_key = f"filter_pending_{column}"
        applied = normalize_selection_key(applied_key, values)
        normalize_selection_key(pending_key, values, default=applied)
        filter_specs.append((column, values, applied_key, pending_key))

    with st.sidebar.expander("Filters", expanded=True):
        with st.form("filters_update_form"):
            for column, values, _applied_key, pending_key in filter_specs:
                st.multiselect(column, values, key=pending_key)
            filters_submitted = st.form_submit_button("Update filters")
        if filters_submitted:
            for _column, values, applied_key, pending_key in filter_specs:
                selected = [
                    value
                    for value in st.session_state.get(pending_key, [])
                    if value in values
                ]
                st.session_state[applied_key] = selected
        for column, _values, applied_key, _pending_key in filter_specs:
            filters[column] = st.session_state.get(applied_key, [])

    st.sidebar.header("Projection")
    limit_genuine = st.sidebar.toggle("Limit Genuine rows", value=False)
    genuine_limit = st.sidebar.number_input(
        "Max Genuine rows", min_value=100, max_value=50000, value=5000, step=100
    )
    method = st.sidebar.radio(
        "Method", ["PCA", "t-SNE", "UMAP", "LDA"], horizontal=True
    )
    scale_features = st.sidebar.toggle("Standardize features", value=True)
    perplexity = st.sidebar.slider(
        "t-SNE perplexity", min_value=2, max_value=80, value=30
    )
    umap_neighbors = st.sidebar.slider(
        "UMAP neighbors", min_value=2, max_value=200, value=15
    )
    umap_min_dist = st.sidebar.slider(
        "UMAP min distance", min_value=0.0, max_value=0.99, value=0.10, step=0.01
    )
    random_state = st.sidebar.number_input(
        "Random seed", min_value=0, max_value=99999, value=42, step=1
    )
    max_plot_rows = st.sidebar.number_input(
        "Max plot rows", min_value=1000, max_value=50000, value=5000, step=1000
    )

    st.sidebar.header("Plot")
    if "fullscreen_plot" not in st.session_state:
        st.session_state["fullscreen_plot"] = True
    fullscreen_plot = st.sidebar.toggle(
        "Full screen plot",
        key="fullscreen_plot",
        help="Use a full-width, taller projection plot. Turn off for plot + inspector split view.",
    )
    color_options = ["merged_class", group_column] + [
        c for c in cats if c != group_column
    ]
    pred_color_label = "model pred > threshold"
    if prediction_column:
        pred_threshold = st.sidebar.slider(
            "Prediction threshold", min_value=0.0, max_value=1.0, value=0.50, step=0.01
        )
        color_options.append(pred_color_label)
    else:
        pred_threshold = 0.50
    color_selection = st.sidebar.selectbox("Color by", options=color_options)
    color_column = (
        PRED_BUCKET_COLUMN if color_selection == pred_color_label else color_selection
    )
    color_title = (
        f"model pred > {pred_threshold:.2f}"
        if color_column == PRED_BUCKET_COLUMN
        else color_column
    )
    symbol_options = ["None"] + cats
    symbol_column = st.sidebar.selectbox("Symbol by", options=symbol_options)
    facet_options = ["None"] + cats
    facet_column = st.sidebar.selectbox("Facet by", options=facet_options)
    render_mode = st.sidebar.selectbox("Render mode", options=["Auto", "WebGL", "SVG"])
    columns = df.columns.tolist()
    model_key = str(st.session_state.get("active_feature_model_key") or "")
    image_candidates = (
        "path",
        "absolute_ori_path",
        "absolute_ocr_path",
        "ori_path",
        "ocr_path",
    )
    image_default_column = configured_image_column_for_model(
        model_key, columns
    ) or find_default_column(columns, image_candidates)
    image_default = (
        columns.index(image_default_column) + 1
        if image_default_column in columns
        else 0
    )
    image_column = st.sidebar.selectbox(
        "Image path column", options=["None"] + columns, index=image_default
    )

    return {
        "selected_features": selected_features,
        "item_id_column": item_id_column,
        "group_column": group_column,
        "merge_rules": merge_rules,
        "filters": filters,
        "method": method,
        "scale_features": scale_features,
        "perplexity": perplexity,
        "umap_neighbors": umap_neighbors,
        "umap_min_dist": float(umap_min_dist),
        "random_state": int(random_state),
        "max_plot_rows": int(max_plot_rows),
        "limit_genuine": limit_genuine,
        "genuine_limit": int(genuine_limit),
        "color_column": color_column,
        "symbol_column": None if symbol_column == "None" else symbol_column,
        "facet_column": None if facet_column == "None" else facet_column,
        "image_column": None if image_column == "None" else image_column,
        "prediction_column": prediction_column,
        "prediction_threshold": float(pred_threshold),
        "render_mode": render_mode,
        "fullscreen_plot": fullscreen_plot,
        "color_title": color_title,
    }


def filter_merged_class(df: pd.DataFrame) -> pd.DataFrame:
    if "merged_class" not in df.columns:
        return df

    values = sorted(df["merged_class"].dropna().astype(str).unique().tolist())
    if not values:
        return df

    st.sidebar.header("Merged Class")
    applied_key = "merged_class_filter_applied"
    pending_key = "merged_class_filter_pending"
    applied = normalize_selection_key(applied_key, values)
    normalize_selection_key(pending_key, values, default=applied)

    with st.sidebar.form("merged_class_update_form"):
        st.multiselect("Visible merged classes", values, key=pending_key)
        submitted = st.form_submit_button("Update classes")
    if submitted:
        st.session_state[applied_key] = [
            value for value in st.session_state.get(pending_key, []) if value in values
        ]

    selected = st.session_state.get(applied_key, [])
    if not selected:
        return df.iloc[0:0]
    return df[df["merged_class"].astype(str).isin(selected)]


def filter_hidden_rows(df: pd.DataFrame) -> pd.DataFrame:
    hidden = st.session_state.get("hidden_row_indexes", set())
    if not hidden:
        return df
    return df[~df.index.isin(hidden)]
