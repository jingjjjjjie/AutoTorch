"""Streamlit view for exploring feature spaces."""
from __future__ import annotations

import os
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, quote, unquote, urlparse

os.environ.setdefault("MPLCONFIGDIR", "/tmp/autotorch_feature_visualization_mpl")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
VENDOR_PATH = Path(__file__).resolve().parent / "vendor"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if VENDOR_PATH.exists() and str(VENDOR_PATH) not in sys.path:
    sys.path.append(str(VENDOR_PATH))
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/autotorch_feature_visualization_numba")

try:
    from advanced_visualization.core.settings import configured_path, load_settings
    from advanced_visualization.core.config import gradcam_artifact_root
    from advanced_visualization.core.gradcam_cache import gradcam_file_index
    from advanced_visualization.core.images import image_cache_digests
except Exception:
    configured_path = None
    gradcam_artifact_root = None
    gradcam_file_index = None
    image_cache_digests = None
    load_settings = None

try:
    from umap import UMAP
    UMAP_IMPORT_ERROR = None
except Exception as exc:
    UMAP = None
    UMAP_IMPORT_ERROR = exc

FEATURE_PREFIXES = ("feature_", "feat_", "embedding_", "emb_")
FEATURE_PATTERN = re.compile(r"^(feature|feat|embedding|emb)[_-]?\d+$", re.IGNORECASE)
PREDICTION_PATTERN = re.compile(r"(pred|prob|score)", re.IGNORECASE)
MAX_TSNE_ROWS = 5000
MAX_UMAP_ROWS = 50000
IMAGE_PROXY_PORT = int(os.environ.get("AUTOTORCH_IMAGE_PROXY_PORT", "8765"))
IMAGE_PROXY_ALLOWED_ROOTS = tuple(
    Path(path).resolve()
    for path in os.environ.get("AUTOTORCH_IMAGE_PROXY_ROOTS", "/routine_data:/mnt3:/mnt4:/mnt5:/app:/home/jingjie/AutoTorch").split(":")
    if path
)
DEFAULT_METADATA_COLUMNS = ("Recapture_Subclass", "Data_Identity", "Quality_Issue")
PRED_BUCKET_COLUMN = "__model_pred_bucket"
PRED_DISPLAY_COLUMN = "__model_pred"


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
    if load_settings is None or configured_path is None:
        return []
    sources: list[tuple[str, Path, str]] = []
    for model in load_settings().models:
        if not model.enabled or not model.feature_csv.strip():
            continue
        sources.append((f"{model.key} - features", configured_path(model.feature_csv), model.key))
    return sources


def default_csv_paths() -> list[Path]:
    multi_paths = os.environ.get("AUTOTORCH_FEATURE_CSVS")
    single_path = os.environ.get("AUTOTORCH_FEATURE_CSV")
    raw_paths = multi_paths or single_path
    if not raw_paths:
        return []
    separator = os.pathsep if os.pathsep in raw_paths else ","
    return [Path(raw_path.strip()).expanduser() for raw_path in raw_paths.split(separator) if raw_path.strip()]


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


def infer_feature_columns(df: pd.DataFrame) -> list[str]:
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    prefix_matches = [
        column
        for column in numeric_columns
        if str(column).lower().startswith(FEATURE_PREFIXES) or FEATURE_PATTERN.match(str(column))
    ]
    if not prefix_matches:
        prefixes = ", ".join(FEATURE_PREFIXES)
        raise ValueError(f"No numeric feature columns found. Expected columns prefixed with one of: {prefixes}")
    return prefix_matches


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


def synced_multiselect(container, label: str, values: list[str], key: str) -> list[str]:
    previous = st.session_state.get(key)
    if previous is not None:
        selected = [value for value in previous if value in values]
        if not selected:
            selected = values
        st.session_state[key] = selected
        return container.multiselect(label, values, key=key)
    return container.multiselect(label, values, default=values, key=key)


def normalize_selection_key(key: str, values: list[str], default: Optional[list[str]] = None) -> list[str]:
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
        raise ValueError("At least three rows with complete numeric features are required.")

    matrix = features.to_numpy(dtype=np.float32)
    if scale_features:
        matrix = StandardScaler().fit_transform(matrix)

    if method == "PCA":
        reducer = PCA(n_components=2, random_state=random_state)
        coords = reducer.fit_transform(matrix)
        subtitle = f"Explained variance: {reducer.explained_variance_ratio_[0]:.1%}, {reducer.explained_variance_ratio_[1]:.1%}"
    elif method == "t-SNE":
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
    elif method == "UMAP":
        if UMAP is None:
            message = "UMAP requires a working umap-learn installation. Install advanced_visualization/requirements.txt."
            if UMAP_IMPORT_ERROR is not None:
                message = f"{message} Import error: {UMAP_IMPORT_ERROR}"
            raise ValueError(message)
        if len(matrix) > MAX_UMAP_ROWS:
            raise ValueError(f"UMAP is limited to {MAX_UMAP_ROWS} selected rows for responsiveness.")
        safe_neighbors = min(max(2, umap_neighbors), max(2, len(matrix) - 1))
        reducer = UMAP(
            n_components=2,
            n_neighbors=safe_neighbors,
            min_dist=umap_min_dist,
            metric="euclidean",
            random_state=random_state,
        )
        coords = reducer.fit_transform(matrix)
        subtitle = f"Neighbors: {safe_neighbors}; min_dist: {umap_min_dist:.2f}"
    elif method == "LDA":
        labels = df.loc[valid_mask, lda_target_column].astype(str)
        class_count = labels.nunique()
        if class_count < 2:
            raise ValueError("LDA requires at least two classes after filtering.")
        max_components = min(matrix.shape[1], class_count - 1)
        if max_components < 1:
            raise ValueError("LDA needs at least one feature dimension and two classes.")
        reducer = LinearDiscriminantAnalysis(n_components=min(2, max_components))
        lda_coords = reducer.fit_transform(matrix, labels)
        if lda_coords.ndim == 1:
            lda_coords = lda_coords.reshape(-1, 1)
        if lda_coords.shape[1] == 1:
            coords = np.column_stack([lda_coords[:, 0], np.zeros(len(lda_coords), dtype=lda_coords.dtype)])
            subtitle = f"Target: {lda_target_column}; 1 discriminant axis for {class_count} classes"
        else:
            coords = lda_coords[:, :2]
            subtitle = f"Target: {lda_target_column}; 2 discriminant axes for {class_count} classes"
    else:
        raise ValueError(f"Unknown projection method: {method}")

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


def image_proxy_url(path_value) -> str:
    if pd.isna(path_value):
        return ""
    return f"/image?path={quote(str(path_value))}"


@st.cache_data(show_spinner=False)
def cached_gradcam_index(root: str, method: str = "", modified_ns: int = 0) -> dict[str, str]:
    if gradcam_file_index is None:
        return {}
    return gradcam_file_index(root, method=method)


def prepared_gradcam_url(path_value, model_key: str, method: str = "gradcam") -> str:
    path = prepared_gradcam_path(path_value, model_key, method=method)
    return image_proxy_url(path) if path else ""


def prepared_gradcam_path(path_value, model_key: str, method: str = "gradcam") -> Optional[Path]:
    if not model_key or pd.isna(path_value) or gradcam_artifact_root is None or image_cache_digests is None:
        return None
    root = gradcam_artifact_root(model_key)
    if root is None or not root.exists():
        return None
    index = cached_gradcam_index(str(root), method, root.stat().st_mtime_ns)
    suffixes = ("_gradcampp_logit", "_gradcampp") if method in {"gradcam++", "gradcampp"} else ("_gradcam_logit", "_gradcam")
    for digest in image_cache_digests(path_value):
        gradcam_path = index.get(digest)
        if gradcam_path and any(marker in Path(gradcam_path).name for marker in suffixes):
            return Path(gradcam_path)
        for marker in suffixes:
            candidate = root / f"{digest}{marker}.png"
            if candidate.is_file():
                return candidate
    return None


def active_prediction_column(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        column
        for column in df.select_dtypes(include=[np.number]).columns
        if PREDICTION_PATTERN.search(str(column)) and column not in infer_feature_columns(df)
    ]
    if len(candidates) == 1:
        return candidates[0]
    generated_candidates = [column for column in candidates if str(column).endswith("_pred")]
    return generated_candidates[-1] if generated_candidates else None


def configured_image_column_for_model(model_key: str, columns: list[str]) -> Optional[str]:
    if not model_key or load_settings is None:
        return None
    for model in load_settings().models:
        if model.key == model_key and model.image_column in columns:
            return model.image_column
    return None


def is_allowed_image_path(path: Path) -> bool:
    try:
        resolved = path.expanduser().resolve()
    except OSError:
        return False
    if not resolved.is_file():
        return False
    if resolved.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
        return False
    return any(resolved == root or root in resolved.parents for root in IMAGE_PROXY_ALLOWED_ROOTS)


def send_file_response(handler: BaseHTTPRequestHandler, path: Path, content_type: str, cache_control: str) -> None:
    try:
        data = path.read_bytes()
    except OSError:
        handler.send_error(404)
        return

    handler.send_response(200)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(data)))
    handler.send_header("Cache-Control", cache_control)
    handler.end_headers()
    handler.wfile.write(data)


class ImageProxyHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        if parsed.path == "/gradcam":
            self.send_error(410, "Interactive Grad-CAM generation is disabled. Use prepared Grad-CAM files.")
            return

        if parsed.path == "/prepared-gradcam":
            model_key = query.get("model", [""])[0]
            raw_image_path = query.get("path", [""])[0]
            if not model_key or not raw_image_path:
                self.send_error(404)
                return
            method = query.get("method", ["gradcam"])[0]
            gradcam_path = prepared_gradcam_path(unquote(raw_image_path), model_key, method=method)
            if gradcam_path is None or not is_allowed_image_path(gradcam_path):
                self.send_error(404)
                return
            send_file_response(self, gradcam_path.expanduser().resolve(), "image/png", "public, max-age=3600")
            return

        if parsed.path != "/image":
            self.send_error(404)
            return

        raw_path = query.get("path", [""])[0]
        image_path = Path(unquote(raw_path))
        if not is_allowed_image_path(image_path):
            self.send_error(404)
            return

        resolved = image_path.expanduser().resolve()
        content_type = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
            ".bmp": "image/bmp",
        }.get(resolved.suffix.lower(), "application/octet-stream")

        send_file_response(self, resolved, content_type, "public, max-age=3600")

    def log_message(self, format: str, *args) -> None:
        return


def ensure_image_proxy() -> None:
    if st.session_state.get("image_proxy_started"):
        return
    try:
        server = ThreadingHTTPServer(("0.0.0.0", IMAGE_PROXY_PORT), ImageProxyHandler)
    except OSError:
        # A previous hot-reload may already have the proxy bound; keep rendering.
        st.session_state["image_proxy_started"] = True
        return

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    st.session_state["image_proxy_started"] = True


def sidebar_controls(df: pd.DataFrame) -> dict:
    st.sidebar.header("Data")
    selected_features = infer_feature_columns(df)
    st.sidebar.caption(f"Using {len(selected_features):,} auto-detected feature columns.")
    prediction_column = active_prediction_column(df)
    if prediction_column:
        st.sidebar.caption(f"Using model prediction: {prediction_column}")

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
                selected = [value for value in st.session_state.get(pending_key, []) if value in values]
                st.session_state[applied_key] = selected
        for column, _values, applied_key, _pending_key in filter_specs:
            filters[column] = st.session_state.get(applied_key, [])

    st.sidebar.header("Projection")
    limit_genuine = st.sidebar.toggle("Limit Genuine rows", value=False)
    genuine_limit = st.sidebar.number_input("Max Genuine rows", min_value=100, max_value=50000, value=5000, step=100)
    method = st.sidebar.radio("Method", ["PCA", "t-SNE", "UMAP", "LDA"], horizontal=True)
    scale_features = st.sidebar.toggle("Standardize features", value=True)
    perplexity = st.sidebar.slider("t-SNE perplexity", min_value=2, max_value=80, value=30)
    umap_neighbors = st.sidebar.slider("UMAP neighbors", min_value=2, max_value=200, value=15)
    umap_min_dist = st.sidebar.slider("UMAP min distance", min_value=0.0, max_value=0.99, value=0.10, step=0.01)
    random_state = st.sidebar.number_input("Random seed", min_value=0, max_value=99999, value=42, step=1)
    max_plot_rows = st.sidebar.number_input("Max plot rows", min_value=1000, max_value=50000, value=5000, step=1000)

    st.sidebar.header("Plot")
    if "fullscreen_plot" not in st.session_state:
        st.session_state["fullscreen_plot"] = True
    fullscreen_plot = st.sidebar.toggle(
        "Full screen plot",
        key="fullscreen_plot",
        help="Use a full-width, taller projection plot. Turn off for plot + inspector split view.",
    )
    color_options = ["merged_class", group_column] + [c for c in cats if c != group_column]
    pred_color_label = "model pred > threshold"
    if prediction_column:
        pred_threshold = st.sidebar.slider("Prediction threshold", min_value=0.0, max_value=1.0, value=0.50, step=0.01)
        color_options.append(pred_color_label)
    else:
        pred_threshold = 0.50
    color_selection = st.sidebar.selectbox("Color by", options=color_options)
    color_column = PRED_BUCKET_COLUMN if color_selection == pred_color_label else color_selection
    color_title = f"model pred > {pred_threshold:.2f}" if color_column == PRED_BUCKET_COLUMN else color_column
    symbol_options = ["None"] + cats
    symbol_column = st.sidebar.selectbox("Symbol by", options=symbol_options)
    facet_options = ["None"] + cats
    facet_column = st.sidebar.selectbox("Facet by", options=facet_options)
    render_mode = st.sidebar.selectbox("Render mode", options=["Auto", "WebGL", "SVG"])
    columns = df.columns.tolist()
    model_key = str(st.session_state.get("active_feature_model_key") or "")
    image_candidates = ("path", "absolute_ori_path", "absolute_ocr_path", "ori_path", "ocr_path")
    image_default_column = configured_image_column_for_model(model_key, columns) or find_default_column(columns, image_candidates)
    image_default = columns.index(image_default_column) + 1 if image_default_column in columns else 0
    image_column = st.sidebar.selectbox("Image path column", options=["None"] + columns, index=image_default)

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


def limit_plot_rows(df: pd.DataFrame, max_rows: int, random_state: int) -> pd.DataFrame:
    if max_rows <= 0 or len(df) <= max_rows:
        return df
    sampled = df.sample(n=max_rows, random_state=random_state).sort_index()
    sampled.attrs.update(df.attrs)
    sampled.attrs["plot_sampling_note"] = f"Random-sampled plot rows from {len(df):,} to {max_rows:,}."
    return sampled


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


def add_prediction_columns(df: pd.DataFrame, prediction_column: Optional[str], threshold: float) -> pd.DataFrame:
    if not prediction_column or prediction_column not in df.columns:
        return df

    prediction = pd.to_numeric(df[prediction_column], errors="coerce")
    df[PRED_DISPLAY_COLUMN] = prediction
    df[PRED_BUCKET_COLUMN] = np.select(
        [prediction.isna(), prediction.gt(threshold)],
        ["missing pred", f"pred > {threshold:.2f}"],
        default=f"pred <= {threshold:.2f}",
    )
    return df


def render_summary(df: pd.DataFrame, projected: pd.DataFrame, group_column: str, feature_count: int) -> None:
    metric_cols = st.columns(4)
    metric_cols[0].metric("Rows selected", f"{len(projected):,}", delta=f"from {len(df):,}")
    metric_cols[1].metric("Feature dimensions", f"{feature_count:,}")
    metric_cols[2].metric("Merged classes", projected["merged_class"].nunique())
    metric_cols[3].metric("Original groups", projected[group_column].nunique())


def count_table(series: pd.Series) -> pd.DataFrame:
    counts = series.fillna("Missing").astype(str).value_counts().rename_axis("class").reset_index(name="count")
    total = counts["count"].sum()
    counts["percent"] = (counts["count"] / total * 100).round(1) if total else 0.0
    return counts


def render_sidebar_counts(projected: pd.DataFrame, group_column: str) -> None:
    st.sidebar.header("Counts")
    st.sidebar.caption(f"Shown points: {len(projected):,}")

    with st.sidebar.expander(group_column, expanded=True):
        st.dataframe(count_table(projected[group_column]), hide_index=True, width="stretch", height=220)

    if group_column != "merged_class" and "merged_class" in projected.columns:
        with st.sidebar.expander("Merged class", expanded=False):
            st.dataframe(count_table(projected["merged_class"]), hide_index=True, width="stretch", height=220)


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
        custom_data = point.get("customdata") or point.get("custom_data") or point.get("customData")
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
        help_text = "Full screen plot" if not fullscreen_enabled else "Exit full screen plot"
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
    plot_df["__hover_item"] = plot_df[item_column].astype(str) if item_column in plot_df.columns else plot_df.index.astype(str)
    plot_df["__hover_class"] = plot_df["merged_class"].astype(str) if "merged_class" in plot_df.columns else ""
    plot_df["__hover_group"] = (
        plot_df[controls["group_column"]].astype(str) if controls["group_column"] in plot_df.columns else ""
    )
    plot_df["__hover_pred"] = (
        plot_df[PRED_DISPLAY_COLUMN].map(lambda value: f"{value:.6f}" if pd.notna(value) else "")
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


def render_client_side_workspace(projected: pd.DataFrame, controls: dict) -> None:
    ensure_image_proxy()
    render_plot_toolbar()
    plot_df = projected.copy()
    item_column = controls["item_id_column"]
    image_column = controls["image_column"]
    model_key = str(st.session_state.get("active_feature_model_key") or "")
    plot_df["__row_index"] = plot_df.index.astype(str)
    plot_df["__hover_item"] = plot_df[item_column].astype(str) if item_column in plot_df.columns else plot_df.index.astype(str)
    plot_df["__hover_class"] = plot_df["merged_class"].astype(str) if "merged_class" in plot_df.columns else ""
    plot_df["__hover_group"] = (
        plot_df[controls["group_column"]].astype(str) if controls["group_column"] in plot_df.columns else ""
    )
    plot_df["__hover_pred"] = (
        plot_df[PRED_DISPLAY_COLUMN].map(lambda value: f"{value:.6f}" if pd.notna(value) else "")
        if PRED_DISPLAY_COLUMN in plot_df.columns
        else ""
    )
    if image_column is not None and image_column in plot_df.columns:
        plot_df["__image_path"] = plot_df[image_column].fillna("").astype(str)
    else:
        plot_df["__image_path"] = ""

    custom_columns = [
        "__row_index",
        "__hover_item",
        "__hover_class",
        "__hover_group",
        "__image_path",
        "__hover_pred",
    ]
    if controls["render_mode"] == "Auto":
        plot_render_mode = "webgl" if len(plot_df) > 2000 else "svg"
    else:
        plot_render_mode = controls["render_mode"].lower()

    plot_height = 880 if controls["fullscreen_plot"] else 720
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
        height=plot_height,
        opacity=0.82,
        render_mode=plot_render_mode,
    )
    fig.update_traces(
        marker=dict(size=8, line=dict(width=0.4, color="rgba(255,255,255,0.45)")),
        hovertemplate=(
            "<b>%{customdata[1]}</b><br>"
            "class=%{customdata[2]}<br>"
            "group=%{customdata[3]}<br>"
            "pred=%{customdata[5]}<br>"
            "x=%{x:.4f}<br>"
            "y=%{y:.4f}"
            "<extra></extra>"
        ),
    )
    fig.update_layout(
        dragmode="pan",
        legend_title_text=controls.get("color_title", controls["color_column"]),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            itemwidth=30,
        ),
        paper_bgcolor="#202124",
        plot_bgcolor="#202124",
        font=dict(color="#f2f3f5"),
        margin=dict(l=8, r=8, t=78, b=8),
        xaxis_title=None,
        yaxis_title=None,
    )

    div_id = "autotorch_feature_plot"
    plot_html = pio.to_html(
        fig,
        include_plotlyjs=True,
        full_html=False,
        div_id=div_id,
        config={"displayModeBar": True, "responsive": True},
    )
    layout_class = "workspace fullscreen" if controls["fullscreen_plot"] else "workspace split"
    component_height = plot_height + 80
    gradcam_action_buttons = """
          <button id="gradcamButton" title="Show prepared Grad-CAM for the selected point" disabled>Grad-CAM</button>
          <button id="gradcamPlusButton" title="Show prepared Grad-CAM++ for the selected point" disabled>Grad-CAM++</button>
    """
    html = f"""
    <style>
      html, body {{
        margin: 0;
        background: #202124;
        color: #f2f3f5;
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        overflow: auto;
      }}
      .workspace {{
        display: grid;
        gap: 14px;
        width: 100%;
      }}
      .workspace.fullscreen {{
        grid-template-columns: minmax(0, 1fr) 380px;
        align-items: start;
      }}
      .workspace.split {{
        grid-template-columns: minmax(0, 1fr) 330px;
        align-items: start;
      }}
      .plot-wrap {{
        min-width: 0;
      }}
      .inspector {{
        border: 1px solid rgba(255,255,255,0.14);
        border-radius: 8px;
        background: #2b2d31;
        padding: 12px;
        max-height: {plot_height}px;
        overflow: auto;
        box-sizing: border-box;
      }}
      .workspace.fullscreen .inspector {{
        max-width: none;
      }}
      .inspector h3 {{
        margin: 0 0 8px 0;
        font-size: 16px;
        font-weight: 650;
      }}
      .muted {{
        color: rgba(242,243,245,0.68);
        font-size: 12px;
        margin-bottom: 10px;
        overflow-wrap: anywhere;
      }}
      .selected-id {{
        font-size: 13px;
        margin-bottom: 8px;
        overflow-wrap: anywhere;
      }}
      .kv {{
        display: grid;
        grid-template-columns: 64px 1fr;
        gap: 4px 8px;
        font-size: 12px;
        margin-bottom: 10px;
      }}
      .kv div:nth-child(odd) {{
        color: rgba(242,243,245,0.62);
      }}
      .inspector-actions {{
        display: flex;
        gap: 8px;
        margin: 10px 0;
        flex-wrap: wrap;
      }}
      .inspector-actions button {{
        border: 1px solid rgba(255,255,255,0.18);
        border-radius: 999px;
        background: #34363b;
        color: #f2f3f5;
        cursor: pointer;
        font-size: 12px;
        padding: 6px 10px;
      }}
      .inspector-actions button:hover {{
        background: #3d4046;
      }}
      #selectedImage {{
        display: none;
        width: 280px;
        max-width: 100%;
        max-height: 300px;
        object-fit: contain;
        border-radius: 6px;
        border: 1px solid rgba(255,255,255,0.14);
        background: #34363b;
      }}
      .workspace.fullscreen #selectedImage {{
        width: 340px;
        max-height: 500px;
      }}
      .image-wrap {{
        position: relative;
        display: inline-block;
        max-width: 100%;
      }}
      .image-fullscreen {{
        display: none;
        position: absolute;
        top: 8px;
        right: 8px;
        width: 34px;
        height: 34px;
        border-radius: 999px;
        border: 1px solid rgba(255,255,255,0.28);
        background: rgba(32,33,36,0.88);
        color: #f2f3f5;
        cursor: pointer;
        font-size: 17px;
        line-height: 1;
      }}
      .image-wrap.has-image .image-fullscreen {{
        display: inline-grid;
        place-items: center;
      }}
      .fullscreen-viewer {{
        position: fixed;
        inset: 0;
        z-index: 9999;
        display: none;
        align-items: center;
        justify-content: center;
        overflow: hidden;
        background: rgba(0,0,0,0.96);
      }}
      .fullscreen-viewer.open {{
        display: flex;
      }}
      .fullscreen-viewer img {{
        max-width: 94vw;
        max-height: 92vh;
        object-fit: contain;
        transform-origin: center center;
        cursor: grab;
        user-select: none;
        -webkit-user-drag: none;
      }}
      .fullscreen-viewer img.dragging {{
        cursor: grabbing;
      }}
      .fullscreen-tools {{
        position: fixed;
        top: 14px;
        right: 14px;
        display: flex;
        gap: 8px;
        z-index: 10000;
      }}
      .fullscreen-tools button {{
        width: 38px;
        height: 38px;
        border-radius: 999px;
        border: 1px solid rgba(255,255,255,0.26);
        background: rgba(20,20,20,0.82);
        color: #ffffff;
        cursor: pointer;
        font-size: 16px;
      }}
      .fullscreen-tools button:hover {{
        background: rgba(45,45,45,0.92);
      }}
      .gradcam-wrap {{
        display: none;
        margin-top: 12px;
      }}
      .gradcam-title {{
        color: rgba(242,243,245,0.72);
        font-size: 12px;
        margin: 0 0 6px 0;
      }}
      #gradcamImage {{
        display: none;
        width: 280px;
        max-width: 100%;
        max-height: 300px;
        object-fit: contain;
        border-radius: 6px;
        border: 1px solid rgba(255,255,255,0.14);
        background: #34363b;
      }}
      .workspace.fullscreen #gradcamImage {{
        width: 340px;
        max-height: 500px;
      }}
      @media (max-width: 980px) {{
        .workspace.fullscreen,
        .workspace.split {{
          grid-template-columns: 1fr;
        }}
        .inspector {{
          max-height: none;
        }}
      }}
    </style>
    <div class="{layout_class}">
      <div class="plot-wrap">{plot_html}</div>
      <aside class="inspector">
        <h3>Item Inspector</h3>
        <div class="muted">Click a point to update this preview without reloading Streamlit.</div>
        <div class="selected-id" id="selectedItem">No point selected.</div>
        <div class="kv">
          <div>Class</div><div id="selectedClass">-</div>
          <div>Group</div><div id="selectedGroup">-</div>
          <div>Pred</div><div id="selectedPred">-</div>
          <div>Row</div><div id="selectedRow">-</div>
          <div>Image</div><div id="selectedPath">-</div>
        </div>
        <div class="inspector-actions">
          <button id="fadePoint" title="Make selected point transparent">Fade point</button>
          <button id="resetFades" title="Restore faded points">Reset fades</button>
          {gradcam_action_buttons}
        </div>
        <div class="muted" id="gradcamStatus"></div>
        <div class="image-wrap" id="imageWrap">
          <img id="selectedImage" alt="Selected item image" />
          <button class="image-fullscreen" id="imageFullscreen" title="Fullscreen image">⛶</button>
        </div>
        <div class="gradcam-wrap" id="gradcamWrap">
          <div class="gradcam-title" id="gradcamTitle">Grad-CAM</div>
          <img id="gradcamImage" alt="Grad-CAM overlay" />
        </div>
      </aside>
    </div>
    <div class="fullscreen-viewer" id="fullscreenViewer">
      <div class="fullscreen-tools">
        <button id="zoomOut" title="Zoom out">-</button>
        <button id="zoomReset" title="Reset zoom">1:1</button>
        <button id="zoomIn" title="Zoom in">+</button>
        <button id="viewerClose" title="Close">×</button>
      </div>
      <img id="fullscreenImage" alt="Fullscreen selected image" />
    </div>
    <script>
      const plot = document.getElementById({div_id!r});
      const activeModelKey = {model_key!r};
      const selectedItem = document.getElementById("selectedItem");
      const selectedClass = document.getElementById("selectedClass");
      const selectedGroup = document.getElementById("selectedGroup");
      const selectedPred = document.getElementById("selectedPred");
      const selectedRow = document.getElementById("selectedRow");
      const selectedPath = document.getElementById("selectedPath");
      const selectedImage = document.getElementById("selectedImage");
      const gradcamImage = document.getElementById("gradcamImage");
      const gradcamWrap = document.getElementById("gradcamWrap");
      const gradcamButton = document.getElementById("gradcamButton");
      const gradcamPlusButton = document.getElementById("gradcamPlusButton");
      const gradcamTitle = document.getElementById("gradcamTitle");
      const gradcamStatus = document.getElementById("gradcamStatus");
      const imageWrap = document.getElementById("imageWrap");
      const imageFullscreen = document.getElementById("imageFullscreen");
      const fullscreenViewer = document.getElementById("fullscreenViewer");
      const fullscreenImage = document.getElementById("fullscreenImage");
      const zoomOut = document.getElementById("zoomOut");
      const zoomReset = document.getElementById("zoomReset");
      const zoomIn = document.getElementById("zoomIn");
      const viewerClose = document.getElementById("viewerClose");
      const fadePoint = document.getElementById("fadePoint");
      const resetFades = document.getElementById("resetFades");
      let activePoint = null;
      let activeImagePath = "";
      let activeGradcamUrl = "";
      let activeGradcamPlusUrl = "";
      let imageScale = 1;
      let imageOffsetX = 0;
      let imageOffsetY = 0;
      let dragStart = null;
      const fadedPoints = new Set();
      const originalMarkers = new Map();

      function traceLength(trace) {{
        if (trace.x && trace.x.length !== undefined) {{
          return trace.x.length;
        }}
        return 0;
      }}

      function cloneValue(value) {{
        if (value === undefined) {{
          return undefined;
        }}
        return JSON.parse(JSON.stringify(value));
      }}

      function pointKey(curveNumber, pointNumber) {{
        return `${{curveNumber}}:${{pointNumber}}`;
      }}

      function toRgba(color, alpha) {{
        if (!color) {{
          return `rgba(31, 119, 180, ${{alpha}})`;
        }}
        if (color.startsWith("#")) {{
          const hex = color.replace("#", "");
          const value = hex.length === 3
            ? hex.split("").map(function(part) {{ return part + part; }}).join("")
            : hex;
          const r = parseInt(value.slice(0, 2), 16);
          const g = parseInt(value.slice(2, 4), 16);
          const b = parseInt(value.slice(4, 6), 16);
          return `rgba(${{r}}, ${{g}}, ${{b}}, ${{alpha}})`;
        }}
        const rgbMatch = color.match(/rgba?\\(([^)]+)\\)/);
        if (rgbMatch) {{
          const parts = rgbMatch[1].split(",").slice(0, 3).map(function(part) {{ return part.trim(); }});
          return `rgba(${{parts[0]}}, ${{parts[1]}}, ${{parts[2]}}, ${{alpha}})`;
        }}
        return color;
      }}

      function saveOriginalMarker(traceIndex) {{
        if (originalMarkers.has(traceIndex)) {{
          return;
        }}
        const trace = plot.data[traceIndex];
        originalMarkers.set(traceIndex, {{
          color: cloneValue(trace.marker.color),
          opacity: cloneValue(trace.marker.opacity)
        }});
      }}

      function originalColorForPoint(marker, pointIndex) {{
        const color = marker.color;
        if (Array.isArray(color)) {{
          return color[pointIndex] || color[0] || "#1f77b4";
        }}
        return color || "#1f77b4";
      }}

      function applyFades() {{
        if (!plot || !plot.data) {{
          return;
        }}
        plot.data.forEach(function(trace, traceIndex) {{
          const savedMarker = originalMarkers.get(traceIndex);
          const hasFadedPoint = Array.from(fadedPoints).some(function(key) {{
            return key.startsWith(`${{traceIndex}}:`);
          }});
          if (!savedMarker || !hasFadedPoint) {{
            return;
          }}
          const n = traceLength(trace);
          const colors = [];
          for (let pointIndex = 0; pointIndex < n; pointIndex += 1) {{
            const key = pointKey(traceIndex, pointIndex);
            const alpha = fadedPoints.has(key) ? 0.12 : 0.82;
            colors.push(toRgba(originalColorForPoint(savedMarker, pointIndex), alpha));
          }}
          Plotly.restyle(plot, {{"marker.color": [colors], "marker.opacity": [1]}}, [traceIndex]);
        }});
      }}

      function proxyUrl(path) {{
        if (!path) {{
          return "";
        }}
        if (path.startsWith("http://") || path.startsWith("https://")) {{
          return path;
        }}
        let pageUrl = null;
        try {{
          pageUrl = new URL(document.referrer);
        }} catch (error) {{
          try {{
            pageUrl = new URL(window.parent.location.href);
          }} catch (parentError) {{
            pageUrl = new URL("http://127.0.0.1:8502/");
          }}
        }}
        return `${{pageUrl.protocol}}//${{pageUrl.hostname}}:{IMAGE_PROXY_PORT}${{path}}`;
      }}

      function updateInspector(point) {{
        const data = point.customdata || [];
        activePoint = {{
          curveNumber: point.curveNumber,
          pointNumber: point.pointNumber
        }};
        selectedRow.textContent = data[0] || "-";
        selectedItem.textContent = data[1] || "Selected item";
        selectedClass.textContent = data[2] || "-";
        selectedGroup.textContent = data[3] || "-";
        selectedPred.textContent = data[5] || "-";
        selectedPath.textContent = data[4] || "-";
        activeImagePath = data[4] || "";
        activeGradcamUrl = activeImagePath && activeModelKey
          ? `/prepared-gradcam?method=gradcam&model=${{encodeURIComponent(activeModelKey)}}&path=${{encodeURIComponent(activeImagePath)}}`
          : "";
        activeGradcamPlusUrl = activeImagePath && activeModelKey
          ? `/prepared-gradcam?method=${{encodeURIComponent("gradcam++")}}&model=${{encodeURIComponent(activeModelKey)}}&path=${{encodeURIComponent(activeImagePath)}}`
          : "";
        gradcamImage.removeAttribute("src");
        gradcamImage.style.display = "none";
        gradcamWrap.style.display = "none";
        if (gradcamButton) {{
          gradcamButton.disabled = !activeGradcamUrl;
        }}
        if (gradcamPlusButton) {{
          gradcamPlusButton.disabled = !activeGradcamPlusUrl;
        }}
        gradcamStatus.textContent = activeGradcamUrl ? "Prepared Grad-CAM available." : "No prepared Grad-CAM for this point.";
        if (activeImagePath) {{
          selectedImage.src = proxyUrl(`/image?path=${{encodeURIComponent(activeImagePath)}}`);
          selectedImage.style.display = "block";
          imageWrap.classList.add("has-image");
        }} else {{
          selectedImage.removeAttribute("src");
          selectedImage.style.display = "none";
          imageWrap.classList.remove("has-image");
        }}
      }}

      fadePoint.addEventListener("click", function() {{
        if (!activePoint) {{
          return;
        }}
        saveOriginalMarker(activePoint.curveNumber);
        fadedPoints.add(pointKey(activePoint.curveNumber, activePoint.pointNumber));
        applyFades();
      }});

      resetFades.addEventListener("click", function() {{
        fadedPoints.clear();
        originalMarkers.forEach(function(marker, traceIndex) {{
          Plotly.restyle(
            plot,
            {{"marker.color": [marker.color], "marker.opacity": [marker.opacity]}},
            [traceIndex]
          );
        }});
        originalMarkers.clear();
      }});

      function showPreparedGradcam(url, label) {{
        if (!url) {{
          gradcamStatus.textContent = `No prepared ${{label}} for this point.`;
          return;
        }}
        gradcamStatus.textContent = `Loading prepared ${{label}}...`;
        gradcamTitle.textContent = label;
        gradcamWrap.style.display = "block";
        gradcamImage.style.display = "none";
        gradcamImage.onload = function() {{
          gradcamStatus.textContent = `Prepared ${{label}} loaded.`;
          gradcamImage.style.display = "block";
        }};
        gradcamImage.onerror = function() {{
          gradcamStatus.textContent = `Prepared ${{label}} failed to load.`;
          gradcamImage.style.display = "none";
        }};
        gradcamImage.src = proxyUrl(url);
      }}

      if (gradcamButton) {{
        gradcamButton.addEventListener("click", function() {{
          showPreparedGradcam(activeGradcamUrl, "Grad-CAM");
        }});
      }}
      if (gradcamPlusButton) {{
        gradcamPlusButton.addEventListener("click", function() {{
          showPreparedGradcam(activeGradcamPlusUrl, "Grad-CAM++");
        }});
      }}

      function applyImageTransform() {{
        fullscreenImage.style.transform = `translate(${{imageOffsetX}}px, ${{imageOffsetY}}px) scale(${{imageScale}})`;
      }}

      function resetImageZoom() {{
        imageScale = 1;
        imageOffsetX = 0;
        imageOffsetY = 0;
        applyImageTransform();
      }}

      function closeImageViewer() {{
        fullscreenViewer.classList.remove("open");
        fullscreenImage.removeAttribute("src");
        if (document.fullscreenElement && document.exitFullscreen) {{
          document.exitFullscreen().catch(function() {{}});
        }}
      }}

      function openImageViewer() {{
        if (!selectedImage.src) {{
          return;
        }}
        fullscreenImage.src = selectedImage.src;
        fullscreenViewer.classList.add("open");
        resetImageZoom();
        if (fullscreenViewer.requestFullscreen) {{
          fullscreenViewer.requestFullscreen().catch(function() {{}});
        }}
      }}

      function zoomImage(delta) {{
        imageScale = Math.min(8, Math.max(0.5, imageScale + delta));
        applyImageTransform();
      }}

      imageFullscreen.addEventListener("click", openImageViewer);
      viewerClose.addEventListener("click", closeImageViewer);
      zoomIn.addEventListener("click", function() {{ zoomImage(0.25); }});
      zoomOut.addEventListener("click", function() {{ zoomImage(-0.25); }});
      zoomReset.addEventListener("click", resetImageZoom);

      fullscreenViewer.addEventListener("wheel", function(event) {{
        event.preventDefault();
        zoomImage(event.deltaY < 0 ? 0.2 : -0.2);
      }}, {{ passive: false }});

      fullscreenImage.addEventListener("mousedown", function(event) {{
        dragStart = {{
          x: event.clientX,
          y: event.clientY,
          offsetX: imageOffsetX,
          offsetY: imageOffsetY
        }};
        fullscreenImage.classList.add("dragging");
      }});

      window.addEventListener("mousemove", function(event) {{
        if (!dragStart) {{
          return;
        }}
        imageOffsetX = dragStart.offsetX + event.clientX - dragStart.x;
        imageOffsetY = dragStart.offsetY + event.clientY - dragStart.y;
        applyImageTransform();
      }});

      window.addEventListener("mouseup", function() {{
        dragStart = null;
        fullscreenImage.classList.remove("dragging");
      }});

      document.addEventListener("fullscreenchange", function() {{
        if (!document.fullscreenElement) {{
          fullscreenViewer.classList.remove("open");
        }}
      }});

      if (plot) {{
        plot.on("plotly_click", function(event) {{
          if (event.points && event.points.length) {{
            updateInspector(event.points[0]);
          }}
        }});
      }}
    </script>
    """
    components.html(html, height=component_height, scrolling=True)


def row_display_name(row: pd.Series, item_id_column: str) -> str:
    stable_id = row.get(item_id_column, row.name)
    return f"{row.name} | {stable_id} | {row.get('merged_class', '')}"


def selected_row_from_selector(projected: pd.DataFrame, item_id_column: str, clicked_index: Optional[int]) -> Optional[int]:
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
        format_func=lambda index: row_display_name(projected.loc[index], item_id_column),
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
        st.image(image, caption=str(row.get(item_id_column, row[image_column])), width=320)


streamlit_fragment = getattr(st, "fragment", lambda func: func)


@streamlit_fragment
def render_interactive_workspace(projected: pd.DataFrame, controls: dict) -> None:
    render_client_side_workspace(projected, controls)


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

        filtered = filtered_frame(df, controls["filters"]).copy()
        filtered = limit_genuine_rows(
            filtered,
            group_column=controls["group_column"],
            enabled=controls["limit_genuine"],
            max_rows=controls["genuine_limit"],
            random_state=controls["random_state"],
        ).copy()
        filtered["merged_class"] = apply_merge_mapping(filtered[controls["group_column"]], controls["merge_rules"])
        filtered = add_prediction_columns(
            filtered,
            controls["prediction_column"],
            controls["prediction_threshold"],
        )
        filtered = filter_merged_class(filtered)
        filtered = filter_hidden_rows(filtered)
        projected_source = limit_plot_rows(filtered, controls["max_plot_rows"], controls["random_state"])
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

    render_summary(filtered, projected, controls["group_column"], len(controls["selected_features"]))
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
        uploaded_file = st.file_uploader("Feature CSV", type=["csv"], label_visibility="collapsed")
    if uploaded_file is not None:
        st.session_state["active_feature_csv_path"] = None
        st.session_state["active_feature_model_key"] = ""
        df = load_csv(uploaded_file)

    if df is None:
        st.info("Configure feature_csv paths in Settings, upload a CSV, or launch with AUTOTORCH_FEATURE_CSV=/path/to/items.csv.")
        return

    render_loaded_data(
        df,
        title="UniRepLKNet-T Item Feature Explorer",
        subtitle="Visualize each provided item as one point, then compare collected, printed, batch, source, and merged class subsets.",
    )


if __name__ == "__main__":
    main()
