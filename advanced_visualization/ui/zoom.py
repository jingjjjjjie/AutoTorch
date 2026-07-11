"""Zoomable image component for image-review cards."""
from __future__ import annotations

import html
from typing import Optional

import streamlit.components.v1 as components
from PIL import Image

from advanced_visualization.core.images import image_to_data_uri


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
            gap: 9px;
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
            border: 1px solid rgba(224,238,232,0.14);
            border-radius: 6px;
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
            padding: 3px 8px;
            background: rgba(13,18,17,0.78);
            border: 1px solid rgba(224,238,232,0.16);
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
