"""Image-review Streamlit view for prepared prediction artifacts."""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/autotorch_advanced_visualization_mpl")

import streamlit as st


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from advanced_visualization.ui.components import (
    render_bottomless_controls,
    render_breakdowns,
    render_grid,
    render_load_more,
    render_pager,
    render_summary,
)
from advanced_visualization.ui.styles import inject_css
from advanced_visualization.views.data_source import load_data
from advanced_visualization.views.filters import add_gradcam_columns, apply_all_filters
from advanced_visualization.views.sidebar import sidebar_controls


if os.environ.get("AUTOTORCH_EMBEDDED_STREAMLIT") != "1":
    st.set_page_config(
        page_title="Advanced Visualization",
        page_icon=".",
        layout="wide",
        initial_sidebar_state="expanded",
    )

def render_loaded_data(df, title: str = "Advanced Visualization", subtitle: str = "Bottomless image review for subclass patterns, confidence failures, and Grad-CAM comparison.") -> None:
    inject_css()
    st.markdown(
        f"""
        <div class="app-hero">
          <h1>{title}</h1>
          <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    controls = sidebar_controls(df)
    filtered = apply_all_filters(df, controls, caption=st.sidebar.caption, include_gradcam=controls["only_prepared_gradcam"])
    render_summary(df, filtered, controls)
    render_breakdowns(filtered, controls)

    if filtered.empty:
        st.warning("No rows match the current filters.")
        return

    if controls["browse_mode"] == "Bottomless scroll":
        visible = render_bottomless_controls(len(filtered), controls["page_size"])
        visible_df = add_gradcam_columns(filtered.iloc[:visible], controls, caption=st.sidebar.caption)
        render_grid(visible_df, controls, start_index=1)
        render_load_more(len(filtered), controls["page_size"])
    else:
        start, end = render_pager(len(filtered), controls["page_size"])
        page_df = add_gradcam_columns(filtered.iloc[start:end], controls, caption=st.sidebar.caption)
        render_grid(page_df, controls, start_index=start + 1)


def main() -> None:
    inject_css()
    df = load_data()
    if df is None:
        st.info("Upload a CSV or configure the prediction CSV and model artifacts in Settings.")
        return

    render_loaded_data(df)


if __name__ == "__main__":
    main()
