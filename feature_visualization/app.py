"""Streamlit app for exploring UniRepLKNet feature spaces."""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


VENDOR_PATH = Path(__file__).resolve().parent / "vendor"
if VENDOR_PATH.exists() and str(VENDOR_PATH) not in sys.path:
    sys.path.append(str(VENDOR_PATH))

try:
    from streamlit_plotly_events import plotly_events
except ImportError:
    plotly_events = None

FEATURE_PATTERN = re.compile(r"^(feature|feat|embedding|emb)[_-]?\d+$", re.IGNORECASE)
PREDICTION_PATTERN = re.compile(r"(_pred_|_result)", re.IGNORECASE)
MAX_TSNE_ROWS = 5000
DEFAULT_METADATA_COLUMNS = ("Recapture_Subclass", "Data_Identity")
LIGHT_SEQUENCE = ["#1f7a8c", "#bf5b17", "#4c6f2f", "#7b2cbf", "#b42318", "#0f766e", "#6b5b00"]
DARK_SEQUENCE = ["#4cc9f0", "#f9a03f", "#9be564", "#c77dff", "#ff6b6b", "#57cc99", "#f7d774"]


st.set_page_config(
    page_title="UniRepLKNet-T Item Feature Explorer",
    page_icon=".",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def inject_css(theme: str) -> None:
    dark = theme == "Dark"
    app_background = "#0d1211" if dark else "#f7faf9"
    text = "#edf7f4" if dark else "#18211f"
    sidebar_background = "#111817" if dark else "#ffffff"
    metric_background = "rgba(30, 41, 39, 0.92)" if dark else "rgba(255,255,255,0.82)"
    border = "rgba(255,255,255,0.14)" if dark else "rgba(16,32,29,0.12)"
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: {app_background};
            color: {text};
        }}
        section[data-testid="stSidebar"] {{
            background: {sidebar_background};
            color: {text};
        }}
        section[data-testid="stSidebar"] * {{
            border-color: {border} !important;
        }}
        div[data-testid="stMetric"] {{
            background: {metric_background};
            border: 1px solid {border};
            border-radius: 8px;
            padding: 0.75rem 0.9rem;
        }}
        .block-container {{
            padding-top: 1rem;
            padding-bottom: 2rem;
            padding-left: 0.75rem;
            padding-right: 0.75rem;
            max-width: none;
        }}
        h1, h2, h3 {{
            letter-spacing: 0;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def load_csv(uploaded_file) -> pd.DataFrame:
    return pd.read_csv(uploaded_file, low_memory=False)


def load_default_csv() -> Optional[pd.DataFrame]:
    default_path = os.environ.get("AUTOTORCH_FEATURE_CSV")
    if not default_path:
        return None
    path = Path(default_path).expanduser()
    if not path.exists():
        st.sidebar.error(f"Default CSV does not exist: {path}")
        return None
    st.sidebar.caption(f"Loaded default CSV: {path}")
    return pd.read_csv(path, low_memory=False)


def infer_feature_columns(df: pd.DataFrame) -> list[str]:
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    pattern_matches = [column for column in numeric_columns if FEATURE_PATTERN.match(str(column))]
    prediction_matches = [column for column in numeric_columns if PREDICTION_PATTERN.search(str(column))]
    return pattern_matches or prediction_matches or numeric_columns


def categorical_columns(df: pd.DataFrame, feature_columns: list[str], metadata_columns: list[str]) -> list[str]:
    excluded = set(feature_columns)
    columns = []
    for column in metadata_columns:
        if column in excluded:
            continue
        if column not in df.columns:
            continue
        unique_count = df[column].nunique(dropna=True)
        if df[column].dtype == "object" or unique_count <= min(80, max(12, len(df) // 5)):
            columns.append(column)
    return columns


def parse_merge_mapping(raw_mapping: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for line_number, raw_line in enumerate(raw_mapping.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        if "=" not in line:
            raise ValueError(f"Merge rule line {line_number} is missing '='.")
        merged_name, values = line.split("=", 1)
        merged_name = merged_name.strip()
        if not merged_name:
            raise ValueError(f"Merge rule line {line_number} has an empty merged class name.")
        source_values = [value.strip() for value in values.split(",") if value.strip()]
        if not source_values:
            raise ValueError(f"Merge rule line {line_number} has no source values.")
        for value in source_values:
            mapping[value] = merged_name
    return mapping


def apply_merge_mapping(series: pd.Series, raw_mapping: str) -> pd.Series:
    if not raw_mapping.strip():
        return series.astype(str)
    mapping = parse_merge_mapping(raw_mapping)
    return series.astype(str).map(lambda value: mapping.get(value, value))


@st.cache_data(show_spinner=False)
def project_features(
    df: pd.DataFrame,
    feature_columns: tuple[str, ...],
    method: str,
    scale_features: bool,
    perplexity: int,
    random_state: int,
) -> pd.DataFrame:
    features = df.loc[:, list(feature_columns)].apply(pd.to_numeric, errors="coerce")
    valid_mask = features.notna().all(axis=1)
    features = features.loc[valid_mask]

    if len(features) < 3:
        raise ValueError("At least three rows with complete numeric features are required.")

    matrix = features.to_numpy(dtype=np.float32)
    if scale_features:
        matrix = StandardScaler().fit_transform(matrix)

    if method == "PCA":
        reducer = PCA(n_components=2, random_state=random_state)
        coords = reducer.fit_transform(matrix)
        subtitle = f"Explained variance: {reducer.explained_variance_ratio_[0]:.1%}, {reducer.explained_variance_ratio_[1]:.1%}"
    else:
        if len(matrix) > MAX_TSNE_ROWS:
            raise ValueError(f"t-SNE is limited to {MAX_TSNE_ROWS} selected rows for responsiveness.")
        safe_perplexity = min(perplexity, max(2, len(matrix) - 1))
        reducer = TSNE(
            n_components=2,
            perplexity=safe_perplexity,
            init="pca",
            learning_rate="auto",
            random_state=random_state,
        )
        coords = reducer.fit_transform(matrix)
        subtitle = f"Perplexity: {safe_perplexity}"

    projected = df.loc[valid_mask].copy()
    projected["x"] = coords[:, 0]
    projected["y"] = coords[:, 1]
    projected.attrs["projection_note"] = subtitle
    return projected


def find_default_column(columns: list[str], candidates: tuple[str, ...]) -> Optional[str]:
    lowered = {str(column).lower(): column for column in columns}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    return columns[0] if columns else None


def image_from_path(path_value) -> Optional[Image.Image]:
    if pd.isna(path_value):
        return None
    path = Path(str(path_value)).expanduser()
    if not path.exists() or not path.is_file():
        return None
    try:
        return Image.open(path).convert("RGB")
    except OSError:
        return None


def sidebar_controls(df: pd.DataFrame, theme: str) -> dict:
    st.sidebar.header("Data")
    inferred_features = infer_feature_columns(df)
    if len(inferred_features) > 50:
        feature_mode = st.sidebar.radio(
            "Feature columns",
            ["All inferred features", "Manual subset"],
            help=f"{len(inferred_features):,} inferred numeric feature columns found.",
        )
        if feature_mode == "All inferred features":
            selected_features = inferred_features
            st.sidebar.caption(f"Using {len(selected_features):,} inferred feature columns.")
        else:
            selected_features = st.sidebar.multiselect(
                "Manual feature subset",
                options=df.columns.tolist(),
                default=inferred_features[: min(32, len(inferred_features))],
                help="Numeric columns used to compute the 2D projection.",
            )
    else:
        selected_features = st.sidebar.multiselect(
            "Feature columns",
            options=df.columns.tolist(),
            default=inferred_features,
            help="Numeric columns used to compute the 2D projection.",
        )

    available_metadata = [column for column in df.columns if column not in selected_features]
    default_metadata = [column for column in DEFAULT_METADATA_COLUMNS if column in available_metadata]
    if not default_metadata:
        default_metadata = [find_default_column(available_metadata, ("class", "sample_type", "source", "label", "batch"))]
        default_metadata = [column for column in default_metadata if column is not None]
    metadata_columns = st.sidebar.multiselect(
        "Metadata columns",
        options=available_metadata,
        default=default_metadata,
        help="Keep this small for large CSVs. These columns become group/filter/color options.",
    )

    cats = categorical_columns(df, selected_features, metadata_columns)
    if not cats:
        raise ValueError("At least one categorical metadata column is required for grouping.")

    id_candidates = ("id", "uuid")
    id_default_column = find_default_column(df.columns.tolist(), id_candidates)
    id_default = df.columns.tolist().index(id_default_column) if id_default_column in df.columns else 0
    item_id_column = st.sidebar.selectbox("Item identifier", options=df.columns.tolist(), index=id_default)

    default_group = find_default_column(cats, ("recapture_subclass", "class", "sample_type", "source", "label", "batch"))
    group_column = st.sidebar.selectbox("Primary class/group", options=cats, index=cats.index(default_group) if default_group in cats else 0)

    merge_rules = st.sidebar.text_area(
        "Merge classes",
        placeholder="colour printed=collected colour printed,printed colour prod\nmix and match=mix,match",
        help="One rule per line. Values not listed are kept unchanged.",
    )

    with st.sidebar.expander("Filters", expanded=True):
        filters = {}
        for column in cats:
            values = sorted(df[column].dropna().astype(str).unique().tolist())
            if not values or len(values) > 120:
                continue
            chosen = st.multiselect(column, values, default=values, key=f"filter_{column}")
            filters[column] = chosen

    st.sidebar.header("Projection")
    limit_genuine = st.sidebar.toggle("Limit Genuine rows", value=True)
    genuine_limit = st.sidebar.number_input("Max Genuine rows", min_value=100, max_value=50000, value=5000, step=100)
    method = st.sidebar.radio("Method", ["PCA", "t-SNE"], horizontal=True)
    scale_features = st.sidebar.toggle("Standardize features", value=True)
    perplexity = st.sidebar.slider("t-SNE perplexity", min_value=2, max_value=80, value=30)
    random_state = st.sidebar.number_input("Random seed", min_value=0, max_value=99999, value=42, step=1)

    st.sidebar.header("Plot")
    fullscreen_plot = st.sidebar.toggle("Wide plot layout", value=True)
    color_column = st.sidebar.selectbox("Color by", options=["merged_class", group_column] + [c for c in cats if c != group_column])
    symbol_options = ["None"] + cats
    symbol_column = st.sidebar.selectbox("Symbol by", options=symbol_options)
    facet_options = ["None"] + cats
    facet_column = st.sidebar.selectbox("Facet by", options=facet_options)
    render_mode = st.sidebar.selectbox("Render mode", options=["Auto", "WebGL", "SVG"])
    image_candidates = ("path", "absolute_ori_path", "absolute_ocr_path", "ori_path", "ocr_path")
    image_default_column = find_default_column(df.columns.tolist(), image_candidates)
    image_default = df.columns.tolist().index(image_default_column) + 1 if image_default_column in df.columns else 0
    image_column = st.sidebar.selectbox("Image path column", options=["None"] + df.columns.tolist(), index=image_default)

    return {
        "selected_features": selected_features,
        "item_id_column": item_id_column,
        "group_column": group_column,
        "merge_rules": merge_rules,
        "filters": filters,
        "method": method,
        "scale_features": scale_features,
        "perplexity": perplexity,
        "random_state": int(random_state),
        "limit_genuine": limit_genuine,
        "genuine_limit": int(genuine_limit),
        "color_column": color_column,
        "symbol_column": None if symbol_column == "None" else symbol_column,
        "facet_column": None if facet_column == "None" else facet_column,
        "image_column": None if image_column == "None" else image_column,
        "render_mode": render_mode,
        "fullscreen_plot": fullscreen_plot,
        "theme": theme,
    }


def filtered_frame(df: pd.DataFrame, filters: dict[str, list[str]]) -> pd.DataFrame:
    filtered = df.copy()
    for column, allowed in filters.items():
        if not allowed:
            return filtered.iloc[0:0]
        filtered = filtered[filtered[column].astype(str).isin(allowed)]
    return filtered


def limit_genuine_rows(
    df: pd.DataFrame,
    group_column: str,
    enabled: bool,
    max_rows: int,
    random_state: int,
) -> pd.DataFrame:
    if not enabled or group_column not in df.columns:
        return df

    group_values = df[group_column].astype(str).str.lower()
    genuine_mask = group_values.eq("genuine")
    genuine = df[genuine_mask]
    if len(genuine) <= max_rows:
        return df

    sampled_genuine = genuine.sample(n=max_rows, random_state=random_state)
    limited = pd.concat([sampled_genuine, df[~genuine_mask]], axis=0).sort_index()
    limited.attrs["sampling_note"] = f"Random-sampled Genuine from {len(genuine):,} to {max_rows:,} rows."
    return limited


def filter_merged_class(df: pd.DataFrame) -> pd.DataFrame:
    if "merged_class" not in df.columns:
        return df

    values = sorted(df["merged_class"].dropna().astype(str).unique().tolist())
    if not values:
        return df

    st.sidebar.header("Merged Class")
    col_select, col_clear = st.sidebar.columns(2)
    if col_select.button("Select all", key="merged_class_select_all"):
        st.session_state["merged_class_filter"] = values
    if col_clear.button("Clear all", key="merged_class_clear_all"):
        st.session_state["merged_class_filter"] = []

    selected = st.sidebar.multiselect(
        "Visible merged classes",
        options=values,
        default=st.session_state.get("merged_class_filter", values),
        key="merged_class_filter",
    )
    if not selected:
        return df.iloc[0:0]
    return df[df["merged_class"].astype(str).isin(selected)]


def filter_hidden_rows(df: pd.DataFrame) -> pd.DataFrame:
    hidden = st.session_state.get("hidden_row_indexes", set())
    if not hidden:
        return df
    return df[~df.index.isin(hidden)]


def render_summary(df: pd.DataFrame, projected: pd.DataFrame, group_column: str, feature_count: int) -> None:
    metric_cols = st.columns(4)
    metric_cols[0].metric("Rows selected", f"{len(projected):,}", delta=f"from {len(df):,}")
    metric_cols[1].metric("Feature dimensions", f"{feature_count:,}")
    metric_cols[2].metric("Merged classes", projected["merged_class"].nunique())
    metric_cols[3].metric("Original groups", projected[group_column].nunique())


def render_distribution(projected: pd.DataFrame, group_column: str, theme: str) -> None:
    counts = (
        projected.groupby(["merged_class", group_column], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    fig = px.bar(
        counts,
        x="merged_class",
        y="count",
        color=group_column,
        barmode="group",
        template="plotly_dark" if theme == "Dark" else "plotly_white",
        color_discrete_sequence=DARK_SEQUENCE if theme == "Dark" else LIGHT_SEQUENCE,
        height=280,
    )
    fig.update_layout(margin=dict(l=8, r=8, t=12, b=8), xaxis_title=None, yaxis_title="Rows")
    st.plotly_chart(fig, use_container_width=True)


def event_to_row_index(events: list[dict], projected: pd.DataFrame) -> Optional[int]:
    if not events:
        return st.session_state.get("clicked_row_index")

    event = events[0]
    custom_data = event.get("customdata")
    if custom_data:
        try:
            return int(custom_data[0])
        except (TypeError, ValueError):
            pass

    if "x" in event and "y" in event:
        distances = (projected["x"] - event["x"]) ** 2 + (projected["y"] - event["y"]) ** 2
        return int(distances.idxmin())

    return st.session_state.get("clicked_row_index")


def render_projection(projected: pd.DataFrame, controls: dict) -> Optional[int]:
    plot_df = projected.copy()
    plot_df["__row_index"] = plot_df.index.astype(str)
    dimmed = st.session_state.get("dimmed_row_indexes", set())
    plot_df["point_state"] = np.where(plot_df.index.isin(dimmed), "dimmed", "active")
    item_column = controls["item_id_column"]
    plot_df["__hover_item"] = plot_df[item_column].astype(str) if item_column in plot_df.columns else plot_df.index.astype(str)
    plot_df["__hover_class"] = plot_df["merged_class"].astype(str) if "merged_class" in plot_df.columns else ""
    plot_df["__hover_group"] = (
        plot_df[controls["group_column"]].astype(str) if controls["group_column"] in plot_df.columns else ""
    )
    custom_columns = ["__row_index", "__hover_item", "__hover_class", "__hover_group"]
    if controls["render_mode"] == "Auto":
        plot_render_mode = "webgl" if len(plot_df) > 5000 else "svg"
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
        template="plotly_dark" if controls["theme"] == "Dark" else "plotly_white",
        color_discrete_sequence=DARK_SEQUENCE if controls["theme"] == "Dark" else LIGHT_SEQUENCE,
        height=820 if controls["fullscreen_plot"] else 720,
        opacity=0.82,
        render_mode=plot_render_mode,
    )
    line_color = "rgba(255,255,255,0.45)" if controls["theme"] == "Dark" else "rgba(20,30,28,0.35)"
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
        legend_title_text=controls["color_column"],
        paper_bgcolor="#0d1211" if controls["theme"] == "Dark" else "#ffffff",
        plot_bgcolor="#121a18" if controls["theme"] == "Dark" else "#ffffff",
        margin=dict(l=8, r=8, t=34, b=8),
        xaxis_title=None,
        yaxis_title=None,
    )
    if plotly_events is None:
        st.warning("Click-to-image needs streamlit-plotly-events. Falling back to the item selector.")
        st.plotly_chart(fig, use_container_width=True)
        return st.session_state.get("clicked_row_index")

    events = plotly_events(
        fig,
        click_event=True,
        select_event=False,
        hover_event=False,
        override_height=820 if controls["fullscreen_plot"] else 720,
        override_width="100%",
        key=f"projection_{controls['theme']}",
    )
    clicked_index = event_to_row_index(events, projected)
    if clicked_index is not None:
        st.session_state["clicked_row_index"] = clicked_index
    return clicked_index


def row_display_name(row: pd.Series, item_id_column: str) -> str:
    stable_id = row.get(item_id_column, row.name)
    return f"{row.name} | {stable_id} | {row.get('merged_class', '')}"


def render_selected_points(
    projected: pd.DataFrame,
    image_column: Optional[str],
    item_id_column: str,
    clicked_index: Optional[int],
) -> None:
    st.subheader("Item Inspector")
    st.caption("Click a point in the plot to show its source image here.")

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
        st.rerun()
    if action_cols[2].button("Undo all", key="undo_hidden_dimmed"):
        st.session_state["dimmed_row_indexes"] = set()
        st.session_state["hidden_row_indexes"] = set()
        st.session_state.pop("clicked_row_index", None)
        st.rerun()
    compact_columns = [
        column
        for column in [item_id_column, "merged_class", image_column, "x", "y"]
        if column is not None and column in selected.columns
    ]
    st.dataframe(selected[compact_columns], use_container_width=True, height=95)

    if image_column is None or image_column not in selected.columns:
        return

    row = selected.iloc[0]
    image = image_from_path(row[image_column])
    if image is None:
        st.caption(f"No image: {row[image_column]}")
    else:
        st.image(image, caption=str(row.get(item_id_column, row[image_column])), use_column_width=True)


def main() -> None:
    theme = st.sidebar.radio("Theme", ["Light", "Dark"], horizontal=True)
    inject_css(theme)
    st.title("UniRepLKNet-T Item Feature Explorer")
    st.caption("Visualize each provided item as one point, then compare collected, printed, batch, source, and merged class subsets.")

    uploaded_file = st.sidebar.file_uploader("Feature CSV", type=["csv"])
    if uploaded_file is None:
        df = load_default_csv()
    else:
        df = load_csv(uploaded_file)

    if df is None:
        st.info("Upload a CSV, or launch with AUTOTORCH_FEATURE_CSV=/path/to/items.csv.")
        return

    try:
        controls = sidebar_controls(df, theme)
        if len(controls["selected_features"]) < 2:
            st.warning("Select at least two numeric feature columns.")
            return

        filtered = filtered_frame(df, controls["filters"]).copy()
        filtered = limit_genuine_rows(
            filtered,
            group_column=controls["group_column"],
            enabled=controls["limit_genuine"],
            max_rows=controls["genuine_limit"],
            random_state=controls["random_state"],
        ).copy()
        filtered["merged_class"] = apply_merge_mapping(filtered[controls["group_column"]], controls["merge_rules"])
        filtered = filter_merged_class(filtered)
        filtered = filter_hidden_rows(filtered)
        projected = project_features(
            filtered,
            tuple(controls["selected_features"]),
            controls["method"],
            controls["scale_features"],
            controls["perplexity"],
            controls["random_state"],
        )
    except Exception as exc:
        st.error(str(exc))
        return

    render_summary(filtered, projected, controls["group_column"], len(controls["selected_features"]))
    st.caption(projected.attrs.get("projection_note", ""))
    if filtered.attrs.get("sampling_note"):
        st.caption(filtered.attrs["sampling_note"])

    if not controls["fullscreen_plot"]:
        left, right = st.columns([0.72, 0.28], gap="large")
        with left:
            clicked_index = render_projection(projected, controls)
        with right:
            render_distribution(projected, controls["group_column"], controls["theme"])
            render_selected_points(projected, controls["image_column"], controls["item_id_column"], clicked_index)
        return

    clicked_index = render_projection(projected, controls)
    lower_left, lower_right = st.columns([0.45, 0.55], gap="large")
    with lower_left:
        render_distribution(projected, controls["group_column"], controls["theme"])
    with lower_right:
        render_selected_points(projected, controls["image_column"], controls["item_id_column"], clicked_index)


if __name__ == "__main__":
    main()
