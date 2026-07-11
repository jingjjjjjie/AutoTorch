"""Launchable model-specific workspace composed from existing viewers."""
from __future__ import annotations

import html

import pandas as pd
import streamlit as st

from advanced_visualization.views import feature_space, image_review
from advanced_visualization.views.extra_views import layered_gradcam


WORKSPACE_SECTIONS = ("Image review", "Feature space", "Layered Grad-CAM")


def _title(config: dict) -> str:
    label = str(config.get("label") or config.get("model_type") or "Model")
    return label.replace("Grad-CAM Review", "Workspace")


def _description(config: dict) -> str:
    return str(
        config.get("workspace_description")
        or "A launched model-specific clone with image review, feature projection, and configured Grad-CAM layer inspection."
    )


def render(df: pd.DataFrame, source: dict, config: dict) -> None:
    title = html.escape(_title(config))
    description = html.escape(_description(config))
    st.markdown(
        f"""
        <div class="app-hero">
          <h1>{title}</h1>
          <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    section = st.sidebar.radio("Workspace section", WORKSPACE_SECTIONS, key=f"{config.get('model_type', 'model')}_workspace_section")
    source_label = str(source.get("label") or source.get("path") or "source")
    st.caption(f"Workspace source: {source_label}")

    if section == "Image review":
        image_review.render_loaded_data(
            df,
            title=f"{_title(config)} Image Review",
            subtitle="Artifact-specific image review using this workspace CSV.",
        )
        return

    if section == "Feature space":
        feature_space.render_loaded_data(
            df,
            title=f"{_title(config)} Feature Space",
            subtitle="Feature projection for this workspace CSV.",
        )
        return

    layered_gradcam.render(df, source, config)
