"""CSV data source selection for the image-review view."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

from advanced_visualization.core.artifacts import available_data_sources, load_manifest, read_csv


@st.cache_data(show_spinner=False)
def read_csv_from_path(path: str, modified_ns: int) -> pd.DataFrame:
    return read_csv(Path(path))


def load_data() -> Optional[pd.DataFrame]:
    st.sidebar.header("Data")
    sources = available_data_sources()
    existing = [source for source in sources if Path(source["path"]).exists()]

    uploaded = None
    with st.sidebar.expander("Upload CSV override", expanded=False):
        uploaded = st.file_uploader("CSV", type=["csv"], label_visibility="collapsed")

    if uploaded is not None:
        st.session_state["advanced_visualization_active_csv_path"] = None
        st.session_state["advanced_visualization_active_csv_stem"] = Path(uploaded.name).stem
        st.session_state.pop("advanced_visualization_artifact_dir", None)
        st.sidebar.caption(f"Uploaded: {uploaded.name}")
        return pd.read_csv(uploaded, low_memory=False)

    if not sources:
        st.sidebar.error("No data sources are configured in settings.")
        return None

    if not existing:
        st.sidebar.error("Configured CSV paths do not exist.")
        for source in sources:
            st.sidebar.caption(str(source["path"]))
        return None

    labels = [str(source["label"]) for source in existing]
    selected = st.sidebar.selectbox("Model / data source", labels, key="advanced_visualization_data_source")
    source = existing[labels.index(selected)]
    path = Path(source["path"])
    st.session_state["advanced_visualization_active_csv_path"] = str(path)

    artifact_dir = source.get("artifact_dir")
    model_key = str(source.get("model_key") or "")
    manifest = load_manifest(Path(artifact_dir)) if artifact_dir else load_manifest(path.parent)
    if manifest and manifest.prepared_csv.resolve() == path.resolve():
        st.session_state["advanced_visualization_active_csv_stem"] = manifest.model_key
        st.session_state["advanced_visualization_artifact_dir"] = str(manifest.artifact_dir)
        st.sidebar.caption(f"Artifact: {manifest.artifact_dir}")
    elif model_key:
        st.session_state["advanced_visualization_active_csv_stem"] = model_key
        if artifact_dir:
            st.session_state["advanced_visualization_artifact_dir"] = str(artifact_dir)
            st.sidebar.caption(f"Artifact: {artifact_dir}")
        else:
            st.session_state.pop("advanced_visualization_artifact_dir", None)
    else:
        st.session_state["advanced_visualization_active_csv_stem"] = path.stem
        st.session_state.pop("advanced_visualization_artifact_dir", None)

    st.sidebar.caption(str(path))
    return read_csv_from_path(str(path), path.stat().st_mtime_ns)
