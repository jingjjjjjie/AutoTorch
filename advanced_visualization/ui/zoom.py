"""Zoomable image component for image-review cards."""
from __future__ import annotations

import html
from os import PathLike
from typing import Optional

import streamlit.components.v1 as components
from PIL import Image

from advanced_visualization.core.images import (
    DEFAULT_PREVIEW_MAX_SIDE,
    DEFAULT_ZOOM_MAX_SIDE,
    image_path_to_data_uri,
    image_to_data_uri,
)


ImageSource = Image.Image | str | PathLike[str]


def _source_to_data_uri(source: Optional[ImageSource], *, max_side: int = DEFAULT_PREVIEW_MAX_SIDE) -> Optional[str]:
    if source is None:
        return None
    if isinstance(source, Image.Image):
        return image_to_data_uri(source)
    return image_path_to_data_uri(source, max_side=max_side)


def render_zoomable_images(
    images: list[tuple[str, Optional[ImageSource]]],
    *,
    preview_height: int | None = None,
    status_label: str = "",
    status_kind: str = "",
    index_label: str = "",
) -> bool:
    panes = []
    for label, source in images:
        preview_uri = _source_to_data_uri(source, max_side=DEFAULT_PREVIEW_MAX_SIDE)
        zoom_uri = _source_to_data_uri(source, max_side=DEFAULT_ZOOM_MAX_SIDE)
        if preview_uri is None:
            continue
        data_uri = zoom_uri or preview_uri
        label_text = html.escape(label)
        status_text = html.escape(status_label)
        status_class = "fail" if status_kind == "fail" else "pass"
        index_text = html.escape(index_label)
        tile_badges = ""
        if status_text or index_text:
            tile_badges = f"""
              <span class="tile-badges">
                <span class="tile-status {status_class}">{status_text}</span>
                <span class="tile-index">{index_text}</span>
              </span>
            """
        panes.append(
            f"""
            <button class="zoom-thumb" data-src="{data_uri}" data-label="{label_text}" title="Open zoom viewer">
              <img src="{preview_uri}" alt="{label_text}" />
              {tile_badges}
              <span class="tile-label">{label_text}</span>
            </button>
            """
        )
    if not panes:
        return False
    grid_class = "single" if len(panes) == 1 else "split"
    thumb_height = int(preview_height or (360 if len(panes) == 1 else 280))
    height = thumb_height + 4
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
            gap: 5px;
            width: 100%;
          }}
          .zoom-grid.split {{
            grid-template-columns: repeat(2, minmax(0, 1fr));
          }}
          .zoom-thumb {{
            position: relative;
            display: block;
            width: 100%;
            height: {thumb_height}px;
            border: 1px solid rgba(224,238,232,0.14);
            border-radius: 5px;
            padding: 0;
            overflow: hidden;
            background: #0d1211;
            cursor: zoom-in;
            transition: border-color 150ms ease, background 150ms ease;
          }}
          .zoom-thumb:hover {{
            border-color: rgba(92,200,215,0.46);
            background: #101716;
          }}
          .zoom-grid.split .zoom-thumb {{
            height: {thumb_height}px;
          }}
          .zoom-thumb img {{
            width: 100%;
            height: 100%;
            object-fit: contain;
            display: block;
            user-select: none;
            -webkit-user-drag: none;
          }}
          .tile-badges {{
            position: absolute;
            left: 5px;
            right: 5px;
            top: 5px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 5px;
            pointer-events: none;
          }}
          .tile-status,
          .tile-index {{
            min-width: 0;
            border-radius: 999px;
            padding: 2px 6px;
            background: rgba(13,18,17,0.76);
            border: 1px solid rgba(224,238,232,0.16);
            color: rgba(238,246,242,0.86);
            font-size: 10px;
            font-weight: 680;
            line-height: 1.2;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
          }}
          .tile-status.fail {{
            color: #ffe0dc;
            border-color: rgba(255,138,128,0.34);
            background: rgba(80,24,23,0.76);
          }}
          .tile-index {{
            flex: 0 0 auto;
            color: #eafaff;
            border-color: rgba(92,200,215,0.42);
            background: rgba(18,68,74,0.78);
          }}
          .zoom-thumb .tile-label {{
            position: absolute;
            left: 5px;
            bottom: 5px;
            max-width: calc(100% - 10px);
            border-radius: 999px;
            padding: 2px 6px;
            background: rgba(13,18,17,0.78);
            border: 1px solid rgba(224,238,232,0.16);
            color: rgba(238,246,242,0.82);
            font-size: 10px;
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
            background: rgba(0,0,0,0.97);
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
            border: 1px solid rgba(224,238,232,0.24);
            background: rgba(18,24,22,0.88);
            color: #eef6f2;
            cursor: pointer;
            font-size: 15px;
            line-height: 1;
          }}
          .zoom-tools button:hover {{
            border-color: rgba(92,200,215,0.52);
            background: rgba(28,42,38,0.94);
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
