"""Paged image viewer for ID-fraud failure and Grad-CAM analysis."""
from __future__ import annotations

import os
import re
import base64
import hashlib
import html
import io
import sys
import threading
from pathlib import Path
from typing import Optional

os.environ.setdefault("MPLCONFIGDIR", "/tmp/autotorch_advanced_visualization_mpl")

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import torch
import torch.nn.functional as F
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data.idfraud.transforms import build_transform
from models import build_model


IMAGE_COLUMNS = (
    "absolute_ori_path",
    "absolute_ocr_path",
    "path",
    "image_path",
    "ori_path",
    "ocr_path",
)
ID_COLUMNS = ("uuid", "id", "item_id", "sample_id")
SUBCLASS_COLUMNS = ("Recapture_Subclass", "Tamper_Subclass", "subclass", "class", "label")
PREDICTION_PATTERN = re.compile(r"(pred|prob|score|result)", re.IGNORECASE)
GRADCAM_PATTERN = re.compile(r"(grad.?cam|cam|heatmap|overlay)", re.IGNORECASE)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DEFAULT_CSV_DIR = Path("/home/jingjie/AutoTorch/feature_visualization/output")
DEFAULT_GRADCAM_ROOT = DEFAULT_CSV_DIR / "gradcam"
DEFAULT_MEAN = (0.485, 0.456, 0.406)
DEFAULT_STD = (0.229, 0.224, 0.225)
GRADCAM_MODEL_LOCK = threading.Lock()
GRADCAM_MODELS: dict[str, dict] = {}
GRADCAM_CONFIGS = {
    "Ex8point2_UniRepLKNet_T_legacy_v1_512_ori_epoch10_full_features": {
        "checkpoint": Path("/mnt3/repo_and_weights/runs/Ex8point2_UniRepLKNet_T_legacy_v1_512_ori/checkpoints/epoch_10.pt"),
        "model_name": "unireplknet_t",
        "head_type": "legacy_v1",
        "image_size": 512,
        "transform_version": "v1",
    },
    "Ex8point4_UniRepLKNet_B_in22k_legacy_v1_512_crop_epoch7_full_features": {
        "checkpoint": Path("/mnt3/repo_and_weights/runs/Ex8point4_UniRepLKNet_B_in22k_legacy_v1_512_crop/checkpoints/epoch_7.pt"),
        "model_name": "unireplknet_b_in22k",
        "head_type": "legacy_v1",
        "image_size": 512,
        "transform_version": "v1",
    },
    "Ex8point2res1024_moredata_largerbs_UniRepLKNet_T_legacy_v1_1024_ori_epoch11_full_features": {
        "checkpoint": Path("/mnt3/repo_and_weights/runs2/Ex8point2res1024_moredata_largerbs_UniRepLKNet_T_legacy_v1_1024_ori/checkpoints/epoch_11.pt"),
        "model_name": "unireplknet_t",
        "head_type": "legacy_v1",
        "image_size": 1024,
        "transform_version": "v1",
    },
}


def gradcam_artifact_root(config_key: str) -> Optional[Path]:
    config = GRADCAM_CONFIGS.get(config_key)
    if not config:
        return None
    checkpoint = config["checkpoint"]
    if checkpoint.parent.name == "checkpoints":
        return checkpoint.parent.parent / "gradcam"
    return checkpoint.parent / "gradcam"


st.set_page_config(
    page_title="Advanced Visualization",
    page_icon=".",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: #0f1312;
            color: #eef6f2;
        }
        section[data-testid="stSidebar"] {
            background: #141918;
        }
        .block-container {
            max-width: none;
            padding: 1rem 1rem 2rem;
        }
        div[data-testid="stMetric"] {
            background: #18201e;
            border: 1px solid rgba(255,255,255,0.12);
            border-radius: 8px;
            padding: 0.7rem 0.85rem;
        }
        div[data-testid="stButton"] > button {
            border-radius: 8px;
        }
        .viewer-card {
            border: 1px solid rgba(255,255,255,0.12);
            border-radius: 8px;
            padding: 8px;
            background: #141918;
            min-height: 100%;
        }
        .viewer-caption {
            font-size: 12px;
            color: rgba(238,246,242,0.72);
            overflow-wrap: anywhere;
            line-height: 1.25;
            margin-top: 6px;
        }
        .filter-strip {
            display: flex;
            flex-wrap: wrap;
            gap: 4px;
            margin: 6px 0 2px;
        }
        .filter-chip {
            display: inline-flex;
            align-items: center;
            gap: 4px;
            max-width: 100%;
            border-radius: 999px;
            border: 1px solid rgba(255,255,255,0.14);
            background: rgba(238,246,242,0.07);
            padding: 2px 7px;
            font-size: 10.5px;
            line-height: 1.25;
            color: rgba(238,246,242,0.84);
        }
        .filter-key {
            color: rgba(238,246,242,0.56);
        }
        .filter-value {
            min-width: 0;
            overflow-wrap: anywhere;
        }
        .index-badge {
            float: right;
            min-width: 30px;
            text-align: center;
            border-radius: 999px;
            padding: 2px 8px;
            margin: 0 0 6px 6px;
            font-size: 12px;
            font-weight: 650;
            color: #eef6f2;
            background: rgba(76, 201, 240, 0.20);
            border: 1px solid rgba(76, 201, 240, 0.42);
        }
        .status-pill {
            display: inline-block;
            border-radius: 999px;
            padding: 2px 7px;
            font-size: 11px;
            border: 1px solid rgba(255,255,255,0.16);
            margin-right: 4px;
        }
        .fail-pill {
            color: #ffcfcb;
            background: rgba(255, 107, 107, 0.16);
        }
        .pass-pill {
            color: #bdf8d2;
            background: rgba(87, 204, 153, 0.16);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def default_csv_paths() -> list[Path]:
    raw_paths = os.environ.get("AUTOTORCH_ADVANCED_VIS_CSV") or os.environ.get("AUTOTORCH_FEATURE_CSV")
    if not raw_paths:
        return sorted(DEFAULT_CSV_DIR.glob("*.csv")) if DEFAULT_CSV_DIR.exists() else []
    separator = os.pathsep if os.pathsep in raw_paths else ","
    paths: list[Path] = []
    for part in raw_paths.split(separator):
        path = Path(part.strip()).expanduser()
        if not path:
            continue
        if path.is_dir():
            paths.extend(sorted(path.glob("*.csv")))
        else:
            paths.append(path)
    return paths


@st.cache_data(show_spinner=False)
def read_csv_from_path(path: str) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def load_data() -> Optional[pd.DataFrame]:
    st.sidebar.header("Data")
    uploaded = st.sidebar.file_uploader("CSV", type=["csv"])
    if uploaded is not None:
        st.session_state["advanced_visualization_active_csv_path"] = None
        st.session_state["advanced_visualization_active_csv_stem"] = Path(uploaded.name).stem
        return pd.read_csv(uploaded, low_memory=False)

    paths = default_csv_paths()
    if not paths:
        return None

    existing = [path for path in paths if path.exists()]
    if not existing:
        st.sidebar.error(f"Default CSV does not exist: {paths[0]}")
        return None

    labels = [path.name for path in existing]
    selected = st.sidebar.selectbox("Default CSV", labels)
    path = existing[labels.index(selected)]
    st.session_state["advanced_visualization_active_csv_path"] = str(path)
    st.session_state["advanced_visualization_active_csv_stem"] = path.stem
    st.sidebar.caption(str(path))
    return read_csv_from_path(str(path))


def first_existing(columns: list[str], candidates: tuple[str, ...]) -> Optional[str]:
    lowered = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def image_path_columns(df: pd.DataFrame) -> list[str]:
    columns = []
    for column in df.columns:
        lower = str(column).lower()
        if column in IMAGE_COLUMNS or "path" in lower or GRADCAM_PATTERN.search(str(column)):
            columns.append(column)
    return columns


def prediction_columns(df: pd.DataFrame) -> list[str]:
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    return [column for column in numeric_columns if PREDICTION_PATTERN.search(str(column))]


def categorical_columns(df: pd.DataFrame) -> list[str]:
    columns = []
    for column in df.columns:
        unique_count = df[column].nunique(dropna=True)
        if df[column].dtype == "object" or unique_count <= min(120, max(12, len(df) // 8)):
            columns.append(column)
    return columns


def numeric_filter_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=[np.number]).columns.tolist()


def valid_image(path_value) -> Optional[Path]:
    if pd.isna(path_value):
        return None
    path = Path(str(path_value)).expanduser()
    if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
        return None
    return path


def load_image(path_value) -> Optional[Image.Image]:
    path = valid_image(path_value)
    if path is None:
        return None
    try:
        return Image.open(path).convert("RGB")
    except OSError:
        return None


def image_to_data_uri(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=88, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def format_card_value(value) -> str:
    if pd.isna(value):
        return "Missing"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4g}"
    if isinstance(value, (int, np.integer)):
        return f"{int(value)}"
    text = str(value)
    return text if text else "Missing"


def card_filter_tags(row: pd.Series, controls: dict) -> list[tuple[str, str]]:
    tags: list[tuple[str, str]] = []
    for column in controls["categorical_filters"]:
        if column in row.index:
            tags.append((str(column), format_card_value(row[column])))
    for column in controls["numeric_ranges"]:
        if column in row.index:
            tags.append((str(column), format_card_value(row[column])))
    return tags


def render_filter_tags(row: pd.Series, controls: dict) -> None:
    tags = card_filter_tags(row, controls)
    if not tags:
        return
    chips = "".join(
        (
            '<span class="filter-chip">'
            f'<span class="filter-key">{html.escape(label)}</span>'
            f'<span class="filter-value">{html.escape(value)}</span>'
            "</span>"
        )
        for label, value in tags
    )
    st.markdown(f'<div class="filter-strip">{chips}</div>', unsafe_allow_html=True)


def render_zoomable_images(images: list[tuple[str, Optional[Image.Image]]]) -> bool:
    available = [(label, image) for label, image in images if image is not None]
    if not available:
        return False

    panes = []
    for label, image in available:
        label_text = html.escape(label)
        data_uri = image_to_data_uri(image)
        panes.append(
            f"""
            <button class="zoom-thumb" data-src="{data_uri}" data-label="{label_text}" title="Open zoom viewer">
              <img src="{data_uri}" alt="{label_text}" />
              <span>{label_text}</span>
            </button>
            """
        )
    grid_class = "single" if len(available) == 1 else "split"
    height = 250 if len(available) == 1 else 230
    components.html(
        f"""
        <style>
          html, body {{
            margin: 0;
            background: transparent;
            color: #eef6f2;
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          }}
          .zoom-grid {{
            display: grid;
            grid-template-columns: 1fr;
            gap: 8px;
            width: 100%;
          }}
          .zoom-grid.split {{
            grid-template-columns: repeat(2, minmax(0, 1fr));
          }}
          .zoom-thumb {{
            position: relative;
            display: block;
            width: 100%;
            height: 220px;
            border: 1px solid rgba(255,255,255,0.14);
            border-radius: 6px;
            padding: 0;
            overflow: hidden;
            background: #0d1211;
            cursor: zoom-in;
          }}
          .zoom-grid.split .zoom-thumb {{
            height: 190px;
          }}
          .zoom-thumb img {{
            width: 100%;
            height: 100%;
            object-fit: contain;
            display: block;
            user-select: none;
            -webkit-user-drag: none;
          }}
          .zoom-thumb span {{
            position: absolute;
            left: 6px;
            bottom: 6px;
            max-width: calc(100% - 12px);
            border-radius: 999px;
            padding: 2px 7px;
            background: rgba(13,18,17,0.74);
            border: 1px solid rgba(255,255,255,0.16);
            color: rgba(238,246,242,0.82);
            font-size: 11px;
            line-height: 1.2;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
          }}
          .zoom-viewer {{
            position: fixed;
            inset: 0;
            z-index: 2147483647;
            display: none;
            align-items: center;
            justify-content: center;
            overflow: hidden;
            background: rgba(0,0,0,0.96);
          }}
          .zoom-viewer.open {{
            display: flex;
          }}
          .zoom-viewer img {{
            max-width: 94vw;
            max-height: 90vh;
            object-fit: contain;
            transform-origin: center center;
            cursor: grab;
            user-select: none;
            -webkit-user-drag: none;
          }}
          .zoom-viewer img.dragging {{
            cursor: grabbing;
          }}
          .zoom-tools {{
            position: fixed;
            top: 14px;
            right: 14px;
            z-index: 2147483647;
            display: flex;
            gap: 8px;
          }}
          .zoom-tools button {{
            min-width: 38px;
            height: 38px;
            border-radius: 999px;
            border: 1px solid rgba(255,255,255,0.26);
            background: rgba(20,20,20,0.86);
            color: #eef6f2;
            cursor: pointer;
            font-size: 15px;
            line-height: 1;
          }}
          .zoom-title {{
            position: fixed;
            left: 14px;
            top: 18px;
            z-index: 2147483647;
            color: rgba(238,246,242,0.82);
            font-size: 13px;
          }}
        </style>
        <div class="zoom-grid {grid_class}">
          {''.join(panes)}
        </div>
        <div class="zoom-viewer" id="zoomViewer">
          <div class="zoom-title" id="zoomTitle"></div>
          <div class="zoom-tools">
            <button id="zoomOut" title="Zoom out">-</button>
            <button id="zoomReset" title="Reset zoom">1:1</button>
            <button id="zoomIn" title="Zoom in">+</button>
            <button id="zoomClose" title="Close">x</button>
          </div>
          <img id="zoomImage" alt="Zoomed image" />
        </div>
        <script>
          const viewer = document.getElementById("zoomViewer");
          const zoomImage = document.getElementById("zoomImage");
          const zoomTitle = document.getElementById("zoomTitle");
          let scale = 1;
          let offsetX = 0;
          let offsetY = 0;
          let dragStart = null;

          function applyTransform() {{
            zoomImage.style.transform = `translate(${{offsetX}}px, ${{offsetY}}px) scale(${{scale}})`;
          }}

          function resetZoom() {{
            scale = 1;
            offsetX = 0;
            offsetY = 0;
            applyTransform();
          }}

          function zoom(delta) {{
            scale = Math.min(10, Math.max(0.4, scale + delta));
            applyTransform();
          }}

          function openViewer(src, label) {{
            zoomImage.src = src;
            zoomTitle.textContent = label || "";
            viewer.classList.add("open");
            resetZoom();
            if (viewer.requestFullscreen) {{
              viewer.requestFullscreen().catch(function() {{}});
            }}
          }}

          function closeViewer() {{
            viewer.classList.remove("open");
            zoomImage.removeAttribute("src");
            if (document.fullscreenElement && document.exitFullscreen) {{
              document.exitFullscreen().catch(function() {{}});
            }}
          }}

          document.querySelectorAll(".zoom-thumb").forEach(function(button) {{
            button.addEventListener("click", function() {{
              openViewer(button.dataset.src, button.dataset.label);
            }});
          }});
          document.getElementById("zoomIn").addEventListener("click", function() {{ zoom(0.25); }});
          document.getElementById("zoomOut").addEventListener("click", function() {{ zoom(-0.25); }});
          document.getElementById("zoomReset").addEventListener("click", resetZoom);
          document.getElementById("zoomClose").addEventListener("click", closeViewer);

          viewer.addEventListener("wheel", function(event) {{
            event.preventDefault();
            zoom(event.deltaY < 0 ? 0.2 : -0.2);
          }}, {{ passive: false }});

          zoomImage.addEventListener("mousedown", function(event) {{
            dragStart = {{
              x: event.clientX,
              y: event.clientY,
              offsetX: offsetX,
              offsetY: offsetY
            }};
            zoomImage.classList.add("dragging");
          }});

          window.addEventListener("mousemove", function(event) {{
            if (!dragStart) {{
              return;
            }}
            offsetX = dragStart.offsetX + event.clientX - dragStart.x;
            offsetY = dragStart.offsetY + event.clientY - dragStart.y;
            applyTransform();
          }});

          window.addEventListener("mouseup", function() {{
            dragStart = null;
            zoomImage.classList.remove("dragging");
          }});

          document.addEventListener("keydown", function(event) {{
            if (event.key === "Escape") {{
              closeViewer();
            }}
          }});

          document.addEventListener("fullscreenchange", function() {{
            if (!document.fullscreenElement) {{
              viewer.classList.remove("open");
            }}
          }});
        </script>
        """,
        height=height,
    )
    return True


def load_state_dict(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    normalized = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        normalized[key] = value
    model.load_state_dict(normalized, strict=True)


def gradcam_device() -> torch.device:
    requested = os.environ.get("AUTOTORCH_GRADCAM_DEVICE")
    if requested:
        return torch.device(requested)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_gradcam_bundle(config_key: str) -> dict:
    with GRADCAM_MODEL_LOCK:
        if config_key in GRADCAM_MODELS:
            return GRADCAM_MODELS[config_key]

        config = GRADCAM_CONFIGS[config_key]
        checkpoint = config["checkpoint"]
        if not checkpoint.exists():
            raise FileNotFoundError(f"Grad-CAM checkpoint missing: {checkpoint}")

        device = gradcam_device()
        model = build_model(
            model_name=config["model_name"],
            device=device,
            task="classification",
            head_type=config["head_type"],
            freeze_backbone=False,
        )
        load_state_dict(model, checkpoint, device)
        model.eval()
        transform = build_transform(
            image_size=config["image_size"],
            normalize_mean=DEFAULT_MEAN,
            normalize_std=DEFAULT_STD,
            version=config["transform_version"],
        )
        bundle = {
            "model": model,
            "transform": transform,
            "target_layer": model.feature_extractor.stages[-1],
            "device": device,
        }
        GRADCAM_MODELS[config_key] = bundle
        return bundle


def overlay_gradcam(original: Image.Image, cam: np.ndarray) -> Image.Image:
    cam = np.nan_to_num(cam, nan=0.0, posinf=1.0, neginf=0.0)
    cam = np.clip(cam, 0.0, 1.0)
    base = np.asarray(original.convert("RGB"), dtype=np.float32)
    heat = np.zeros_like(base)
    heat[..., 0] = 255.0 * cam
    heat[..., 1] = 210.0 * np.sqrt(cam)
    heat[..., 2] = 28.0 * (1.0 - cam) * cam
    alpha = np.clip(0.18 + 0.55 * cam[..., None], 0.18, 0.65)
    overlay = base * (1.0 - alpha) + heat * alpha
    return Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))


def gradcam_score(model: torch.nn.Module, input_tensor: torch.Tensor) -> torch.Tensor:
    features = model.feature_extractor(input_tensor)
    head_sequence = getattr(model.mlp_head, "fc", None)
    if head_sequence is None:
        head_sequence = getattr(model.mlp_head, "head", None)

    if isinstance(head_sequence, torch.nn.Sequential) and len(head_sequence) > 0:
        layers = list(head_sequence.children())
        if isinstance(layers[-1], torch.nn.Sigmoid):
            score = features
            for layer in layers[:-1]:
                score = layer(score)
            return score.squeeze()

    return model.mlp_head(features).squeeze()


def compute_cam(activation: torch.Tensor, gradient: torch.Tensor) -> torch.Tensor:
    weights = gradient.mean(dim=(2, 3), keepdim=True)
    return torch.relu((weights * activation).sum(dim=1, keepdim=True))


def generate_gradcam(config_key: str, image_path: Path) -> Path:
    if config_key not in GRADCAM_CONFIGS:
        raise ValueError(f"No Grad-CAM model config for {config_key}")
    if not image_path.is_file():
        raise FileNotFoundError(f"Image missing: {image_path}")

    output_root = gradcam_artifact_root(config_key) or (DEFAULT_GRADCAM_ROOT / config_key)
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = gradcam_cache_candidates(output_root, image_path)[0]
    if output_path.exists():
        return output_path

    bundle = load_gradcam_bundle(config_key)
    model = bundle["model"]
    transform = bundle["transform"]
    target_layer = bundle["target_layer"]
    device = bundle["device"]
    activations = {}
    gradients = {}

    def forward_hook(_module, _inputs, output):
        activations["value"] = output

    def backward_hook(_module, _grad_input, grad_output):
        gradients["value"] = grad_output[0]

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)
    try:
        image = Image.open(image_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)
        model.zero_grad(set_to_none=True)
        score = gradcam_score(model, input_tensor).mean()
        score.backward()

        activation = activations["value"].detach()
        gradient = gradients["value"].detach()
        cam = compute_cam(activation, gradient)
        cam = F.interpolate(cam, size=(image.height, image.width), mode="bilinear", align_corners=False)
        cam = cam.squeeze().float()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        tmp_output = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
        overlay_gradcam(image, cam.detach().cpu().numpy()).save(tmp_output, format="PNG", compress_level=3)
        os.replace(tmp_output, output_path)
        return output_path
    finally:
        if "tmp_output" in locals():
            tmp_output.unlink(missing_ok=True)
        forward_handle.remove()
        backward_handle.remove()


@st.cache_data(show_spinner=False)
def gradcam_file_index(root: str) -> dict[str, str]:
    root_path = Path(root).expanduser()
    if not root_path.exists():
        return {}
    index = {}
    for path in root_path.glob("*.png"):
        digest = path.name.split("_", 1)[0]
        if len(digest) != 18:
            continue
        current = index.get(digest)
        if current is None:
            index[digest] = str(path)
            continue
        current_name = Path(current).name
        priority = ("_gradcam_logit", "_gradcampp_logit", "_gradcam", "_gradcampp")
        if priority_index(path.name, priority) < priority_index(current_name, priority):
            index[digest] = str(path)
    return index


def priority_index(name: str, priority: tuple[str, ...]) -> int:
    for index, marker in enumerate(priority):
        if marker in name:
            return index
    return len(priority)


def image_cache_digest(path_value) -> Optional[str]:
    image_path = valid_image(path_value)
    if image_path is None:
        return None
    resolved = image_path.expanduser().resolve()
    try:
        stamp = f"{resolved}:{resolved.stat().st_mtime_ns}"
    except OSError:
        return None
    return hashlib.sha1(stamp.encode("utf-8")).hexdigest()[:18]


def truth_as_bool(series: pd.Series, positive_values: str, numeric_threshold: float) -> pd.Series:
    positive_set = {value.strip().lower() for value in positive_values.split(",") if value.strip()}
    numeric = pd.to_numeric(series, errors="coerce")
    result = numeric.ge(numeric_threshold)
    text = series.astype(str).str.strip().str.lower()
    if positive_set:
        result = result.where(numeric.notna(), text.isin(positive_set))
        result = result | text.isin(positive_set)
    return result.fillna(False).astype(bool)


def add_failure_columns(
    df: pd.DataFrame,
    truth_column: Optional[str],
    prediction_column: Optional[str],
    threshold: float,
    positive_values: str,
    truth_threshold: float,
) -> pd.DataFrame:
    enriched = df.copy()
    if truth_column is None or prediction_column is None:
        enriched["__has_eval"] = False
        enriched["__prediction_score"] = np.nan
        enriched["__predicted_positive"] = False
        enriched["__actual_positive"] = False
        enriched["__is_failure"] = False
        enriched["__failure_type"] = "unscored"
        enriched["__confidence"] = np.nan
        return enriched

    score = pd.to_numeric(enriched[prediction_column], errors="coerce")
    predicted_positive = score.ge(threshold)
    actual_positive = truth_as_bool(enriched[truth_column], positive_values, truth_threshold)
    has_eval = score.notna() & enriched[truth_column].notna()
    failure = predicted_positive.ne(actual_positive) & has_eval

    enriched["__has_eval"] = has_eval
    enriched["__prediction_score"] = score
    enriched["__predicted_positive"] = predicted_positive
    enriched["__actual_positive"] = actual_positive
    enriched["__is_failure"] = failure
    enriched["__confidence"] = np.maximum(score, 1.0 - score)
    enriched["__failure_type"] = np.select(
        [
            ~has_eval,
            predicted_positive & ~actual_positive,
            ~predicted_positive & actual_positive,
            ~failure,
        ],
        ["unscored", "false positive", "false negative", "correct"],
        default="failure",
    )
    return enriched


def apply_failure_mode(df: pd.DataFrame, mode: str, high_conf: float, low_conf: float) -> pd.DataFrame:
    if mode == "All rows":
        return df
    if mode == "Failures only":
        return df[df["__is_failure"]]
    if mode == "Correct only":
        return df[df["__failure_type"].eq("correct")]
    if mode == "False positives":
        return df[df["__failure_type"].eq("false positive")]
    if mode == "False negatives":
        return df[df["__failure_type"].eq("false negative")]
    if mode == "High-confidence failures":
        return df[df["__is_failure"] & df["__confidence"].ge(high_conf)]
    if mode == "Low-confidence failures":
        return df[df["__is_failure"] & df["__confidence"].le(low_conf)]
    return df


def apply_text_search(df: pd.DataFrame, query: str, columns: list[str]) -> pd.DataFrame:
    query = query.strip()
    if not query or not columns:
        return df
    mask = pd.Series(False, index=df.index)
    for column in columns:
        mask |= df[column].astype(str).str.contains(query, case=False, na=False, regex=False)
    return df[mask]


def apply_categorical_filters(df: pd.DataFrame, filters: dict[str, list[str]]) -> pd.DataFrame:
    filtered = df
    for column, values in filters.items():
        if not values:
            return filtered.iloc[0:0]
        filtered = filtered[filtered[column].fillna("Missing").astype(str).isin(values)]
    return filtered


def apply_numeric_ranges(df: pd.DataFrame, ranges: dict[str, tuple[float, float]]) -> pd.DataFrame:
    filtered = df
    for column, (minimum, maximum) in ranges.items():
        values = pd.to_numeric(filtered[column], errors="coerce")
        filtered = filtered[values.between(minimum, maximum, inclusive="both")]
    return filtered


def has_gradcam(row: pd.Series, controls: dict) -> bool:
    return resolve_gradcam_path(row, controls) is not None


def gradcam_roots(active_stem: Optional[str], gradcam_dir: str = "") -> list[Path]:
    roots: list[Path] = []
    if gradcam_dir:
        roots.append(Path(gradcam_dir).expanduser())
    if active_stem:
        artifact_root = gradcam_artifact_root(active_stem)
        if artifact_root is not None:
            roots.append(artifact_root)
        roots.append(DEFAULT_GRADCAM_ROOT / active_stem)

    unique_roots = []
    seen = set()
    for root in roots:
        key = str(root)
        if key not in seen:
            unique_roots.append(root)
            seen.add(key)
    return unique_roots


def add_gradcam_columns(df: pd.DataFrame, controls: dict) -> pd.DataFrame:
    if df.empty:
        enriched = df.copy()
        enriched["__gradcam_path"] = ""
        enriched["__has_gradcam"] = False
        return enriched
    enriched = df.copy()

    paths = None
    image_column = controls["image_column"]
    active_stem = controls.get("active_csv_stem")
    roots = [root for root in gradcam_roots(active_stem, controls.get("gradcam_dir", "")) if root.exists()]
    should_preload = image_column and image_column in enriched.columns and roots
    if should_preload:
        index = {}
        loaded_count = 0
        for root in roots:
            root_index = gradcam_file_index(str(root))
            loaded_count += len(root_index)
            index.update({digest: path for digest, path in root_index.items() if digest not in index})
        digests = enriched[image_column].map(image_cache_digest)
        paths = digests.map(lambda digest: index.get(digest, "") if digest else "")
        st.sidebar.caption(f"Preloaded {loaded_count:,} Grad-CAM files from {len(roots)} folder(s)")

    if paths is None:
        paths = df.apply(lambda row: resolve_gradcam_path(row, controls), axis=1).map(
            lambda path: str(path) if path is not None else ""
        )
    elif controls["gradcam_column"] or controls["gradcam_dir"]:
        missing = paths.eq("")
        if missing.any():
            fallback = enriched[missing].apply(lambda row: resolve_gradcam_path(row, controls), axis=1).map(
                lambda path: str(path) if path is not None else ""
            )
            paths.loc[missing] = fallback

    enriched["__gradcam_path"] = paths.map(lambda path: str(path) if path is not None else "")
    enriched["__has_gradcam"] = enriched["__gradcam_path"].ne("")
    return enriched


def gradcam_cache_candidates(root: Path, image_path: Path) -> list[Path]:
    resolved = image_path.expanduser().resolve()
    try:
        stamp = f"{resolved}:{resolved.stat().st_mtime_ns}"
    except OSError:
        return []
    digest = hashlib.sha1(stamp.encode("utf-8")).hexdigest()[:18]
    return [
        root / f"{digest}_gradcam_logit.png",
        root / f"{digest}_gradcampp_logit.png",
        root / f"{digest}_gradcam.png",
        root / f"{digest}_gradcampp.png",
    ]


def resolve_gradcam_path(row: pd.Series, controls: dict):
    gradcam_column = controls["gradcam_column"]
    gradcam_dir = controls["gradcam_dir"]
    image_column = controls["image_column"]
    if gradcam_column and gradcam_column in row.index:
        path = valid_image(row[gradcam_column])
        if path is not None:
            return path

    if not image_column or image_column not in row.index:
        return None

    image_path = valid_image(row[image_column])
    if image_path is None:
        return None

    active_stem = controls.get("active_csv_stem")
    roots = gradcam_roots(active_stem, gradcam_dir)

    for root in roots:
        for candidate in gradcam_cache_candidates(root, image_path):
            if candidate.is_file():
                return candidate

    root = roots[0] if roots else DEFAULT_GRADCAM_ROOT / str(active_stem or "")
    candidates = [
        root / f"{image_path.stem}.png",
        root / f"{image_path.stem}.jpg",
        root / f"{image_path.stem}_gradcam.png",
        root / f"{image_path.stem}_overlay.png",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def gradcam_for_row(row: pd.Series, controls: dict) -> tuple[Optional[Path], Optional[str]]:
    existing = row.get("__gradcam_path") or resolve_gradcam_path(row, controls)
    if existing:
        return Path(str(existing)), None

    if not controls.get("generate_missing_gradcam"):
        return None, None

    active_stem = controls.get("active_csv_stem")
    image_column = controls.get("image_column")
    if not active_stem or active_stem not in GRADCAM_CONFIGS:
        return None, "No Grad-CAM config for this CSV."
    if not image_column or image_column not in row.index:
        return None, "No image column selected."

    image_path = valid_image(row[image_column])
    if image_path is None:
        return None, "Image file is missing."

    try:
        return generate_gradcam(active_stem, image_path), None
    except Exception as exc:
        return None, str(exc)


def sidebar_controls(df: pd.DataFrame) -> dict:
    columns = df.columns.astype(str).tolist()
    image_columns = image_path_columns(df)
    pred_columns = prediction_columns(df)
    cats = categorical_columns(df)

    id_default = first_existing(columns, ID_COLUMNS) or columns[0]
    image_default = first_existing(columns, IMAGE_COLUMNS)
    subclass_default = first_existing(columns, SUBCLASS_COLUMNS)
    truth_default = "label" if "label" in df.columns else subclass_default
    pred_default = pred_columns[-1] if pred_columns else None
    gradcam_candidates = [column for column in image_columns if GRADCAM_PATTERN.search(str(column))]

    st.sidebar.header("Columns")
    item_id_column = st.sidebar.selectbox("Item ID", columns, index=columns.index(id_default))
    image_column = st.sidebar.selectbox(
        "Image path",
        ["None"] + image_columns,
        index=(image_columns.index(image_default) + 1 if image_default in image_columns else 0),
    )
    subclass_column = st.sidebar.selectbox(
        "Subclass/group",
        ["None"] + cats,
        index=(cats.index(subclass_default) + 1 if subclass_default in cats else 0),
    )
    truth_column = st.sidebar.selectbox(
        "Truth column",
        ["None"] + columns,
        index=(columns.index(truth_default) + 1 if truth_default in columns else 0),
    )
    prediction_column = st.sidebar.selectbox(
        "Prediction score",
        ["None"] + pred_columns,
        index=(pred_columns.index(pred_default) + 1 if pred_default in pred_columns else 0),
    )
    gradcam_column = st.sidebar.selectbox(
        "Grad-CAM path column",
        ["None"] + image_columns,
        index=(image_columns.index(gradcam_candidates[0]) + 1 if gradcam_candidates else 0),
    )
    active_csv_stem = st.session_state.get("advanced_visualization_active_csv_stem")
    default_gradcam_dir = gradcam_artifact_root(active_csv_stem) if active_csv_stem else None
    gradcam_dir = st.sidebar.text_input(
        "Grad-CAM directory",
        value=str(default_gradcam_dir) if default_gradcam_dir and default_gradcam_dir.exists() else "",
        help="Optional fallback: looks for image-stem PNG/JPG files in this directory.",
        key=f"gradcam_dir_{active_csv_stem or 'uploaded'}",
    )

    st.sidebar.header("Failure Logic")
    prediction_threshold = st.sidebar.slider("Prediction threshold", 0.0, 1.0, 0.5, 0.01)
    positive_values = st.sidebar.text_input(
        "Positive truth values",
        value="1,true,positive,fraud,recapture,tamper",
        help="Comma-separated labels treated as positive when truth is not numeric.",
    )
    truth_threshold = st.sidebar.number_input("Numeric truth threshold", value=0.5, step=0.1)
    high_conf = st.sidebar.slider("High confidence >=", 0.0, 1.0, 0.9, 0.01)
    low_conf = st.sidebar.slider("Low confidence <=", 0.0, 1.0, 0.6, 0.01)
    mode = st.sidebar.radio(
        "Failure view",
        [
            "All rows",
            "Failures only",
            "High-confidence failures",
            "Low-confidence failures",
            "False positives",
            "False negatives",
            "Correct only",
        ],
    )

    st.sidebar.header("Advanced Filters")
    search_columns = st.sidebar.multiselect(
        "Search columns",
        columns,
        default=[column for column in [item_id_column, subclass_column] if column and column != "None"],
    )
    text_query = st.sidebar.text_input("Search text")
    default_filter_columns = [column for column in [subclass_column, "Data_Identity"] if column and column != "None" and column in cats]
    filter_columns = st.sidebar.multiselect(
        "Categorical filters",
        cats,
        default=default_filter_columns,
    )
    categorical_filters = {}
    for column in filter_columns:
        values = sorted(df[column].fillna("Missing").astype(str).unique().tolist())
        default_values = values
        categorical_filters[column] = st.sidebar.multiselect(column, values, default=default_values)

    numeric_ranges = {}
    with st.sidebar.expander("Numeric ranges", expanded=False):
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
            selected = st.slider(column, min_value, max_value, (min_value, max_value))
            numeric_ranges[column] = selected

    st.sidebar.header("Paging")
    browse_mode = st.sidebar.radio(
        "Browse mode",
        ["Bottomless scroll", "Pages"],
        index=0,
        horizontal=True,
    )
    view_mode = st.sidebar.radio(
        "Image mode",
        ["Original", "Grad-CAM", "Side-by-side"],
        index=0,
        horizontal=True,
    )
    only_cached_gradcam = st.sidebar.checkbox(
        "Only cached Grad-CAM",
        value=False,
        help="Uses feature_visualization/output/gradcam/<selected_csv_stem>/ automatically.",
        key=f"only_cached_gradcam_{view_mode}",
    )
    generate_missing_gradcam = st.sidebar.checkbox(
        "Generate missing Grad-CAM",
        value=view_mode != "Original",
        help="Generates Grad-CAM for visible cards when no saved overlay exists yet.",
    )
    page_size = st.sidebar.select_slider("Page size", options=[12, 24, 48, 96, 144], value=48)
    columns_per_row = st.sidebar.slider("Columns per row", 2, 8, 6)
    sort_options = ["confidence desc", "confidence asc", "prediction desc", "prediction asc", "row order"]
    sort_by = st.sidebar.selectbox("Sort", sort_options)

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
        "positive_values": positive_values,
        "truth_threshold": truth_threshold,
        "high_conf": high_conf,
        "low_conf": low_conf,
        "mode": mode,
        "search_columns": search_columns,
        "text_query": text_query,
        "categorical_filters": categorical_filters,
        "numeric_ranges": numeric_ranges,
        "browse_mode": browse_mode,
        "view_mode": view_mode,
        "only_cached_gradcam": only_cached_gradcam,
        "generate_missing_gradcam": generate_missing_gradcam,
        "page_size": page_size,
        "columns_per_row": columns_per_row,
        "sort_by": sort_by,
    }


def apply_all_filters(df: pd.DataFrame, controls: dict) -> pd.DataFrame:
    filtered = add_failure_columns(
        df,
        controls["truth_column"],
        controls["prediction_column"],
        controls["prediction_threshold"],
        controls["positive_values"],
        controls["truth_threshold"],
    )
    filtered = apply_failure_mode(filtered, controls["mode"], controls["high_conf"], controls["low_conf"])
    filtered = apply_text_search(filtered, controls["text_query"], controls["search_columns"])
    filtered = apply_categorical_filters(filtered, controls["categorical_filters"])
    filtered = apply_numeric_ranges(filtered, controls["numeric_ranges"])
    filtered = add_gradcam_columns(filtered, controls)
    if controls["only_cached_gradcam"]:
        filtered = filtered[filtered["__has_gradcam"]]

    sort_by = controls["sort_by"]
    if sort_by == "confidence desc":
        filtered = filtered.sort_values("__confidence", ascending=False, na_position="last")
    elif sort_by == "confidence asc":
        filtered = filtered.sort_values("__confidence", ascending=True, na_position="last")
    elif sort_by == "prediction desc":
        filtered = filtered.sort_values("__prediction_score", ascending=False, na_position="last")
    elif sort_by == "prediction asc":
        filtered = filtered.sort_values("__prediction_score", ascending=True, na_position="last")
    return filtered


def render_summary(source: pd.DataFrame, filtered: pd.DataFrame, controls: dict) -> None:
    scored = filtered[filtered["__has_eval"]]
    failures = int(filtered["__is_failure"].sum()) if "__is_failure" in filtered else 0
    failure_rate = failures / len(scored) if len(scored) else 0.0

    cached_gradcam = int(filtered["__has_gradcam"].sum()) if "__has_gradcam" in filtered else 0

    metric_cols = st.columns(6)
    metric_cols[0].metric("Rows shown", f"{len(filtered):,}", delta=f"from {len(source):,}")
    metric_cols[1].metric("Scored rows", f"{len(scored):,}")
    metric_cols[2].metric("Failures", f"{failures:,}", delta=f"{failure_rate:.1%}")
    metric_cols[3].metric("High-conf failures", f"{int((filtered['__is_failure'] & filtered['__confidence'].ge(controls['high_conf'])).sum()):,}")
    metric_cols[4].metric("Low-conf failures", f"{int((filtered['__is_failure'] & filtered['__confidence'].le(controls['low_conf'])).sum()):,}")
    metric_cols[5].metric("Existing Grad-CAM", f"{cached_gradcam:,}")


def page_bounds(total: int, page_size: int) -> tuple[int, int, int]:
    total_pages = max(1, int(np.ceil(total / page_size)))
    page_key = "advanced_visualization_page"
    current = int(st.session_state.get(page_key, 1))
    current = min(max(1, current), total_pages)
    st.session_state[page_key] = current
    start = (current - 1) * page_size
    end = min(start + page_size, total)
    return current, total_pages, start, end


def render_pager(total: int, page_size: int) -> tuple[int, int]:
    current, total_pages, start, end = page_bounds(total, page_size)
    left, middle, right = st.columns([0.18, 0.64, 0.18])
    with left:
        if st.button("Previous", disabled=current <= 1, use_container_width=True):
            st.session_state["advanced_visualization_page"] = current - 1
            st.rerun()
    with middle:
        selected = st.number_input("Page", min_value=1, max_value=total_pages, value=current, step=1)
        if selected != current:
            st.session_state["advanced_visualization_page"] = int(selected)
            st.rerun()
        st.caption(f"Showing {start + 1 if total else 0:,}-{end:,} of {total:,}")
    with right:
        if st.button("Next", disabled=current >= total_pages, use_container_width=True):
            st.session_state["advanced_visualization_page"] = current + 1
            st.rerun()
    return start, end


def visible_count(total: int, batch_size: int) -> int:
    key = "advanced_visualization_visible_count"
    current = int(st.session_state.get(key, batch_size))
    current = min(max(batch_size, current), max(total, batch_size))
    st.session_state[key] = current
    return min(current, total)


def render_bottomless_controls(total: int, batch_size: int) -> int:
    count = visible_count(total, batch_size)
    st.caption(f"Showing 1-{count:,} of {total:,}")
    return count


def render_load_more(total: int, batch_size: int) -> None:
    count = visible_count(total, batch_size)
    if count >= total:
        st.caption(f"All {total:,} rows shown.")
        return

    left, middle, right = st.columns([0.25, 0.50, 0.25])
    with middle:
        if st.button(f"Load next {batch_size}", use_container_width=True):
            st.session_state["advanced_visualization_visible_count"] = min(total, count + batch_size)
            st.rerun()
        st.caption(f"Showing {count:,} of {total:,}")


def row_label(row: pd.Series, controls: dict) -> str:
    item = row.get(controls["item_id_column"], row.name)
    subclass = row.get(controls["subclass_column"], "") if controls["subclass_column"] else ""
    score = row.get("__prediction_score", np.nan)
    confidence = row.get("__confidence", np.nan)
    score_text = f"pred={score:.4f}" if pd.notna(score) else "pred=-"
    conf_text = f"conf={confidence:.3f}" if pd.notna(confidence) else "conf=-"
    return f"{item}\n{subclass}\n{score_text} | {conf_text}"


def render_image_cell(
    row: pd.Series,
    controls: dict,
    *,
    display_index: Optional[int] = None,
) -> None:
    image_column = controls["image_column"]
    original = load_image(row[image_column]) if image_column else None
    view_mode = controls["view_mode"]
    gradcam_error = None
    if view_mode == "Original":
        gradcam_path = row.get("__gradcam_path") or resolve_gradcam_path(row, controls)
    else:
        gradcam_path, gradcam_error = gradcam_for_row(row, controls)
    gradcam = load_image(gradcam_path) if gradcam_path else None

    is_failure = bool(row.get("__is_failure", False))
    pill_class = "fail-pill" if is_failure else "pass-pill"
    pill_text = row.get("__failure_type", "unscored")
    if display_index is not None:
        st.markdown(f'<span class="index-badge">#{display_index}</span>', unsafe_allow_html=True)
    st.markdown(f'<span class="status-pill {pill_class}">{pill_text}</span>', unsafe_allow_html=True)

    if view_mode == "Original":
        if not render_zoomable_images([("Original", original)]):
            st.caption("No image")
    elif view_mode == "Grad-CAM":
        if not render_zoomable_images([("Grad-CAM", gradcam)]):
            st.caption("No Grad-CAM")
            if gradcam_error:
                st.caption(gradcam_error)
    else:
        if not render_zoomable_images([("Original", original), ("Grad-CAM", gradcam)]):
            st.caption("Missing")
        if original is None:
            st.caption("Original missing")
        if gradcam is None:
            st.caption("Grad-CAM missing")
            if gradcam_error:
                st.caption(gradcam_error)

    render_filter_tags(row, controls)
    st.markdown(f'<div class="viewer-caption">{row_label(row, controls)}</div>', unsafe_allow_html=True)
    if gradcam_path:
        st.caption(Path(str(gradcam_path)).name)


def render_grid(page_df: pd.DataFrame, controls: dict, start_index: int = 1) -> None:
    cols_per_row = controls["columns_per_row"]
    rows = list(enumerate(page_df.iterrows(), start=start_index))
    for offset in range(0, len(rows), cols_per_row):
        columns = st.columns(cols_per_row)
        for column, (display_index, (_index, row)) in zip(columns, rows[offset : offset + cols_per_row]):
            with column:
                with st.container(border=True):
                    render_image_cell(row, controls, display_index=display_index)


def render_breakdowns(filtered: pd.DataFrame, controls: dict) -> None:
    with st.expander("Breakdowns", expanded=True):
        columns = st.columns(2)
        with columns[0]:
            st.caption("Failure type")
            counts = filtered["__failure_type"].value_counts(dropna=False).rename_axis("type").reset_index(name="count")
            st.dataframe(counts, hide_index=True, use_container_width=True, height=220)
        with columns[1]:
            subclass_column = controls["subclass_column"]
            if subclass_column and subclass_column in filtered.columns:
                st.caption(f"Failures by {subclass_column}")
                table = (
                    filtered.assign(__failure=filtered["__is_failure"].astype(int))
                    .groupby(subclass_column, dropna=False)
                    .agg(rows=("__failure", "size"), failures=("__failure", "sum"), mean_confidence=("__confidence", "mean"))
                    .reset_index()
                    .sort_values(["failures", "rows"], ascending=False)
                )
                table["failure_rate"] = (table["failures"] / table["rows"]).round(4)
                st.dataframe(table, hide_index=True, use_container_width=True, height=220)


def main() -> None:
    inject_css()
    st.title("Advanced Visualization")
    st.caption("Bottomless image review for subclass patterns, high/low confidence failures, and Grad-CAM comparison.")

    df = load_data()
    if df is None:
        st.info("Upload a CSV or launch with AUTOTORCH_ADVANCED_VIS_CSV=/path/to/file.csv.")
        return

    controls = sidebar_controls(df)
    filtered = apply_all_filters(df, controls)
    render_summary(df, filtered, controls)
    render_breakdowns(filtered, controls)

    if filtered.empty:
        st.warning("No rows match the current filters.")
        return

    if controls["browse_mode"] == "Bottomless scroll":
        visible = render_bottomless_controls(len(filtered), controls["page_size"])
        visible_df = filtered.iloc[:visible]
        render_grid(visible_df, controls, start_index=1)
        render_load_more(len(filtered), controls["page_size"])
    else:
        start, end = render_pager(len(filtered), controls["page_size"])
        page_df = filtered.iloc[start:end]
        render_grid(page_df, controls, start_index=start + 1)


if __name__ == "__main__":
    main()
