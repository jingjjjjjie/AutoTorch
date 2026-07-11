"""Unified visualization entrypoint.

This is the preferred Streamlit entrypoint. It combines the prepared artifact
image-review workflow with the feature-space explorer while keeping their
implementation modules separate.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ["AUTOTORCH_EMBEDDED_STREAMLIT"] = "1"

st.set_page_config(
    page_title="AutoTorch Visualization",
    page_icon=".",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main() -> None:
    view = st.sidebar.radio(
        "Page",
        ["Image review", "Feature space", "Settings"],
    )

    if view == "Image review":
        from advanced_visualization.views import image_review as image_review_app

        image_review_app.main()
        return

    if view == "Feature space":
        from advanced_visualization.views import feature_space as feature_space_app

        feature_space_app.main()
        return

    from advanced_visualization.views import settings as settings_app

    settings_app.main()


if __name__ == "__main__":
    main()
