"""Streamlit view for exploring feature spaces."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

os.environ.setdefault("MPLCONFIGDIR", "/tmp/autotorch_feature_visualization_mpl")

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image

from advanced_visualization.core.dataframe_filters import apply_categorical_filters
from advanced_visualization.core.feature_data import (
    PRED_DISPLAY_COLUMN,
    add_prediction_columns,
    apply_merge_mapping,
    limit_named_group,
    limit_rows,
)
from advanced_visualization.core.images import load_image
from advanced_visualization.core.projection import ProjectionParameters, project_matrix
from advanced_visualization.core.settings import configured_path, load_settings
from advanced_visualization.views.feature_controls import (
    filter_hidden_rows,
    filter_merged_class,
    sidebar_controls,
)


if os.environ.get("AUTOTORCH_EMBEDDED_STREAMLIT") != "1":
    st.set_page_config(
        page_title="UniRepLKNet-T Item Feature Explorer",
        page_icon=".",
        layout="wide",
        initial_sidebar_state="collapsed",
    )


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: #202124 !important;
            color: #f2f3f5 !important;
        }
        section[data-testid="stSidebar"] {
            background: #26282c !important;
            border-right: 1px solid rgba(255,255,255,0.12) !important;
        }
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {
            color: #f2f3f5 !important;
        }
        div[data-testid="stMetric"] {
            background: #2b2d31 !important;
            border: 1px solid rgba(255,255,255,0.14) !important;
            border-radius: 8px;
        }
        .block-container {
            padding-top: 1rem;
            padding-bottom: 2rem;
            padding-left: 0.75rem;
            padding-right: 0.75rem;
            max-width: none;
        }
        h1, h2, h3 {
            letter-spacing: 0;
        }
        div[data-testid="stButton"] > button[kind="secondary"] {
            border-radius: 999px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def load_csv(uploaded_file) -> pd.DataFrame:
    return pd.read_csv(uploaded_file, low_memory=False)


@st.cache_data(show_spinner=False)
def read_feature_csv(path: str, modified_ns: int) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def settings_feature_csv_paths() -> list[tuple[str, Path, str]]:
    sources: list[tuple[str, Path, str]] = []
    for model in load_settings().models:
        if not model.enabled or not model.feature_csv.strip():
            continue
        sources.append(
            (f"{model.key} - features", configured_path(model.feature_csv), model.key)
        )
    return sources


def default_csv_paths() -> list[Path]:
    multi_paths = os.environ.get("AUTOTORCH_FEATURE_CSVS")
    single_path = os.environ.get("AUTOTORCH_FEATURE_CSV")
    raw_paths = multi_paths or single_path
    if not raw_paths:
        return []
    separator = os.pathsep if os.pathsep in raw_paths else ","
    return [
        Path(raw_path.strip()).expanduser()
        for raw_path in raw_paths.split(separator)
        if raw_path.strip()
    ]


def default_feature_sources() -> list[tuple[str, Path, str]]:
    sources = settings_feature_csv_paths()
    sources.extend((f"{path.stem} - env", path, "") for path in default_csv_paths())
    return sources


def load_default_csv() -> Optional[pd.DataFrame]:
    sources = default_feature_sources()
    if not sources:
        st.session_state["active_feature_csv_path"] = None
        return None

    st.sidebar.header("Feature source")
    if len(sources) == 1:
        label, path, model_key = sources[0]
        st.sidebar.caption(label)
    else:
        labels = [label for label, _path, _model_key in sources]
        selected_label = st.sidebar.selectbox("Feature set", labels)
        _label, path, model_key = sources[labels.index(selected_label)]

    if not path.exists():
        st.sidebar.error(f"Feature CSV does not exist: {path}")
        st.session_state["active_feature_csv_path"] = None
        return None
    st.session_state["active_feature_csv_path"] = str(path.resolve())
    st.session_state["active_feature_model_key"] = model_key
    st.sidebar.caption(str(path))
    return read_feature_csv(str(path), path.stat().st_mtime_ns)


@st.cache_data(show_spinner=False)
def project_features(
    df: pd.DataFrame,
    feature_columns: tuple[str, ...],
    method: str,
    scale_features: bool,
    perplexity: int,
    umap_neighbors: int,
    umap_min_dist: float,
    lda_target_column: Optional[str],
    random_state: int,
) -> pd.DataFrame:
    features = df.loc[:, list(feature_columns)].apply(pd.to_numeric, errors="coerce")
    valid_mask = features.notna().all(axis=1)
    if method == "LDA":
        if lda_target_column is None or lda_target_column not in df.columns:
            raise ValueError("LDA requires a valid class/group column.")
        valid_mask &= df[lda_target_column].notna()
    features = features.loc[valid_mask]

    if len(features) < 3:
        raise ValueError(
            "At least three rows with complete numeric features are required."
        )

    labels = (
        df.loc[valid_mask, lda_target_column].astype(str).to_numpy()
        if method == "LDA" and lda_target_column
        else None
    )
    projection = project_matrix(
        features.to_numpy(dtype=np.float32),
        ProjectionParameters(
            method={"PCA": "pca", "t-SNE": "tsne", "UMAP": "umap", "LDA": "lda"}[
                method
            ],
            scale=scale_features,
            perplexity=perplexity,
            umap_neighbors=umap_neighbors,
            umap_min_dist=umap_min_dist,
            random_state=random_state,
        ),
        labels=labels,
    )

    projected = df.loc[valid_mask].copy()
    projected["x"] = projection.values[:, 0]
    projected["y"] = projection.values[:, 1]
    projected.attrs["projection_note"] = projection.note
    return projected


def image_from_path(path_value) -> Optional[Image.Image]:
    return load_image(path_value)


def render_summary(
    df: pd.DataFrame, projected: pd.DataFrame, group_column: str, feature_count: int
) -> None:
    metric_cols = st.columns(4)
    metric_cols[0].metric(
        "Rows selected", f"{len(projected):,}", delta=f"from {len(df):,}"
    )
    metric_cols[1].metric("Feature dimensions", f"{feature_count:,}")
    metric_cols[2].metric("Merged classes", projected["merged_class"].nunique())
    metric_cols[3].metric("Original groups", projected[group_column].nunique())


def count_table(series: pd.Series) -> pd.DataFrame:
    counts = (
        series.fillna("Missing")
        .astype(str)
        .value_counts()
        .rename_axis("class")
        .reset_index(name="count")
    )
    total = counts["count"].sum()
    counts["percent"] = (counts["count"] / total * 100).round(1) if total else 0.0
    return counts


def render_sidebar_counts(projected: pd.DataFrame, group_column: str) -> None:
    st.sidebar.header("Counts")
    st.sidebar.caption(f"Shown points: {len(projected):,}")

    with st.sidebar.expander(group_column, expanded=True):
        st.dataframe(
            count_table(projected[group_column]),
            hide_index=True,
            width="stretch",
            height=220,
        )

    if group_column != "merged_class" and "merged_class" in projected.columns:
        with st.sidebar.expander("Merged class", expanded=False):
            st.dataframe(
                count_table(projected["merged_class"]),
                hide_index=True,
                width="stretch",
                height=220,
            )


def plotly_state_to_row_index(state, projected: pd.DataFrame) -> Optional[int]:
    selection = getattr(state, "selection", None)
    if selection is None and isinstance(state, dict):
        selection = state.get("selection")
    points = getattr(selection, "points", None)
    if points is None and isinstance(selection, dict):
        points = selection.get("points")
    if not points:
        return None

    point = points[0]
    custom_data = getattr(point, "customdata", None)
    if custom_data is None and isinstance(point, dict):
        custom_data = (
            point.get("customdata")
            or point.get("custom_data")
            or point.get("customData")
        )
    if custom_data:
        try:
            return int(custom_data[0])
        except (TypeError, ValueError):
            pass

    point_index = getattr(point, "point_index", None)
    if point_index is None and isinstance(point, dict):
        point_index = point.get("point_index")
        point_index = point.get("pointIndex", point_index)
        point_index = point.get("point_number", point_index)
        point_index = point.get("pointNumber", point_index)
    if point_index is not None and 0 <= int(point_index) < len(projected):
        return int(projected.index[int(point_index)])

    return None


def render_plot_toolbar() -> None:
    toolbar_left, toolbar_right = st.columns([0.94, 0.06])
    with toolbar_right:
        fullscreen_enabled = st.session_state.get("fullscreen_plot", True)
        label = "⛶" if not fullscreen_enabled else "×"
        help_text = (
            "Full screen plot" if not fullscreen_enabled else "Exit full screen plot"
        )
        if st.button(label, help=help_text, key="fullscreen_plot_round_button"):
            st.session_state["fullscreen_plot"] = not fullscreen_enabled
            st.rerun()


def render_projection(projected: pd.DataFrame, controls: dict) -> Optional[int]:
    render_plot_toolbar()
    plot_df = projected.copy()
    plot_df["__row_index"] = plot_df.index.astype(str)
    dimmed = st.session_state.get("dimmed_row_indexes", set())
    plot_df["point_state"] = np.where(plot_df.index.isin(dimmed), "dimmed", "active")
    item_column = controls["item_id_column"]
    plot_df["__hover_item"] = (
        plot_df[item_column].astype(str)
        if item_column in plot_df.columns
        else plot_df.index.astype(str)
    )
    plot_df["__hover_class"] = (
        plot_df["merged_class"].astype(str) if "merged_class" in plot_df.columns else ""
    )
    plot_df["__hover_group"] = (
        plot_df[controls["group_column"]].astype(str)
        if controls["group_column"] in plot_df.columns
        else ""
    )
    plot_df["__hover_pred"] = (
        plot_df[PRED_DISPLAY_COLUMN].map(
            lambda value: f"{value:.6f}" if pd.notna(value) else ""
        )
        if PRED_DISPLAY_COLUMN in plot_df.columns
        else ""
    )
    custom_columns = ["__row_index", "__hover_item", "__hover_class", "__hover_group"]
    if controls["render_mode"] == "Auto":
        plot_render_mode = "webgl" if len(plot_df) > 2000 else "svg"
    else:
        plot_render_mode = controls["render_mode"].lower()

    fig = px.scatter(
        plot_df,
        x="x",
        y="y",
        color=controls["color_column"],
        symbol=controls["symbol_column"],
        facet_col=controls["facet_column"],
        hover_name=item_column if item_column in plot_df.columns else None,
        custom_data=custom_columns,
        template="plotly",
        height=920 if controls["fullscreen_plot"] else 720,
        opacity=0.82,
        render_mode=plot_render_mode,
    )
    line_color = "rgba(255,255,255,0.45)"
    fig.update_traces(
        marker=dict(size=8, line=dict(width=0.4, color=line_color)),
        hovertemplate=(
            "<b>%{customdata[1]}</b><br>"
            "class=%{customdata[2]}<br>"
            "group=%{customdata[3]}<br>"
            "x=%{x:.4f}<br>"
            "y=%{y:.4f}"
            "<extra></extra>"
        ),
    )
    for trace in fig.data:
        if trace.customdata is None:
            continue
        trace.marker.opacity = [
            0.18 if int(custom_data[0]) in dimmed else 0.82
            for custom_data in trace.customdata
        ]
    fig.update_layout(
        dragmode="select",
        legend_title_text=controls.get("color_title", controls["color_column"]),
        paper_bgcolor="#202124",
        plot_bgcolor="#202124",
        font=dict(color="#f2f3f5"),
        margin=dict(l=8, r=8, t=34, b=8),
        xaxis_title=None,
        yaxis_title=None,
    )
    plot_height = 920 if controls["fullscreen_plot"] else 720
    state = st.plotly_chart(
        fig,
        width="stretch",
        height=plot_height,
        key="projection_native_dark",
        theme=None,
        on_select="rerun",
        selection_mode="points",
        config={"displayModeBar": True},
    )
    clicked_index = plotly_state_to_row_index(state, projected)
    if clicked_index is not None:
        st.session_state["clicked_row_index"] = clicked_index
    return clicked_index


def row_display_name(row: pd.Series, item_id_column: str) -> str:
    stable_id = row.get(item_id_column, row.name)
    return f"{row.name} | {stable_id} | {row.get('merged_class', '')}"


def selected_row_from_selector(
    projected: pd.DataFrame, item_id_column: str, clicked_index: Optional[int]
) -> Optional[int]:
    options = projected.index.tolist()
    if not options:
        return None

    if clicked_index in projected.index:
        st.session_state["inspector_selected_index"] = clicked_index

    current = st.session_state.get("inspector_selected_index")
    if current not in options:
        current = options[0]
        st.session_state["inspector_selected_index"] = current

    st.caption("Select a point in the plot, or choose an item here.")
    return st.selectbox(
        "Item",
        options=options,
        format_func=lambda index: row_display_name(
            projected.loc[index], item_id_column
        ),
        key="inspector_selected_index",
    )


def render_selected_points(
    projected: pd.DataFrame,
    image_column: Optional[str],
    item_id_column: str,
    clicked_index: Optional[int],
) -> None:
    st.subheader("Item Inspector")
    st.caption("Select a point in the plot to show its source image here.")
    clicked_index = selected_row_from_selector(projected, item_id_column, clicked_index)

    if clicked_index not in projected.index:
        st.info("No point selected yet.")
        return

    selected = projected.loc[[clicked_index]].copy()
    selected_item = selected.iloc[0].get(item_id_column, clicked_index)
    st.markdown(f"**Selected item:** `{selected_item}`")
    action_cols = st.columns(3)
    dimmed = st.session_state.setdefault("dimmed_row_indexes", set())
    hidden = st.session_state.setdefault("hidden_row_indexes", set())
    if action_cols[0].button("Toggle dark", key=f"toggle_dim_{clicked_index}"):
        if clicked_index in dimmed:
            dimmed.remove(clicked_index)
        else:
            dimmed.add(clicked_index)
        st.rerun()
    if action_cols[1].button("Delete", key=f"hide_{clicked_index}"):
        hidden.add(clicked_index)
        st.session_state.pop("clicked_row_index", None)
        st.session_state.pop("inspector_selected_index", None)
        st.rerun()
    if action_cols[2].button("Undo all", key="undo_hidden_dimmed"):
        st.session_state["dimmed_row_indexes"] = set()
        st.session_state["hidden_row_indexes"] = set()
        st.session_state.pop("clicked_row_index", None)
        st.session_state.pop("inspector_selected_index", None)
        st.rerun()
    compact_columns = [
        column
        for column in [item_id_column, "merged_class", image_column, "x", "y"]
        if column is not None and column in selected.columns
    ]
    st.dataframe(selected[compact_columns], width="stretch", height=95)

    if image_column is None or image_column not in selected.columns:
        return

    row = selected.iloc[0]
    image = image_from_path(row[image_column])
    if image is None:
        st.caption(f"No image: {row[image_column]}")
    else:
        st.image(
            image, caption=str(row.get(item_id_column, row[image_column])), width=320
        )


streamlit_fragment = getattr(st, "fragment", lambda func: func)


@streamlit_fragment
def render_interactive_workspace(projected: pd.DataFrame, controls: dict) -> None:
    clicked_index = render_projection(projected, controls)
    render_selected_points(
        projected,
        controls["image_column"],
        controls["item_id_column"],
        clicked_index,
    )


def render_loaded_data(
    df: pd.DataFrame,
    *,
    title: str = "Feature Space",
    subtitle: str = "Visualize each provided item as one point, then compare metadata, prediction, and class subsets.",
) -> None:
    inject_css()
    st.title(title)
    st.caption(subtitle)

    try:
        controls = sidebar_controls(df)
        if len(controls["selected_features"]) < 2:
            st.warning("Select at least two numeric feature columns.")
            return

        filtered = apply_categorical_filters(df, controls["filters"])
        if controls["limit_genuine"]:
            filtered = limit_named_group(
                filtered,
                group_column=controls["group_column"],
                group_name="Genuine",
                max_rows=controls["genuine_limit"],
                random_state=controls["random_state"],
            )
        filtered["merged_class"] = apply_merge_mapping(
            filtered[controls["group_column"]], controls["merge_rules"]
        )
        filtered = add_prediction_columns(
            filtered,
            controls["prediction_column"],
            controls["prediction_threshold"],
        )
        filtered = filter_merged_class(filtered)
        filtered = filter_hidden_rows(filtered)
        projected_source = limit_rows(
            filtered, controls["max_plot_rows"], controls["random_state"]
        )
        projected = project_features(
            projected_source,
            tuple(controls["selected_features"]),
            controls["method"],
            controls["scale_features"],
            controls["perplexity"],
            controls["umap_neighbors"],
            controls["umap_min_dist"],
            "merged_class" if controls["method"] == "LDA" else None,
            controls["random_state"],
        )
    except Exception as exc:
        st.error(str(exc))
        return

    render_summary(
        filtered,
        projected,
        controls["group_column"],
        len(controls["selected_features"]),
    )
    st.caption(projected.attrs.get("projection_note", ""))
    if filtered.attrs.get("sampling_note"):
        st.caption(filtered.attrs["sampling_note"])
    if projected.attrs.get("plot_sampling_note"):
        st.caption(projected.attrs["plot_sampling_note"])
    render_sidebar_counts(projected, controls["group_column"])

    render_interactive_workspace(projected, controls)


def main() -> None:
    df = load_default_csv()
    uploaded_file = None
    with st.sidebar.expander("Upload feature CSV override", expanded=False):
        uploaded_file = st.file_uploader(
            "Feature CSV", type=["csv"], label_visibility="collapsed"
        )
    if uploaded_file is not None:
        st.session_state["active_feature_csv_path"] = None
        st.session_state["active_feature_model_key"] = ""
        df = load_csv(uploaded_file)

    if df is None:
        st.info(
            "Configure feature_csv paths in Settings, upload a CSV, or launch with AUTOTORCH_FEATURE_CSV=/path/to/items.csv."
        )
        return

    render_loaded_data(
        df,
        title="UniRepLKNet-T Item Feature Explorer",
        subtitle="Visualize each provided item as one point, then compare collected, printed, batch, source, and merged class subsets.",
    )


if __name__ == "__main__":
    main()
