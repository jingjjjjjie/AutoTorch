"""Launch configured model-specific workspaces."""
from __future__ import annotations

from pathlib import Path
from urllib.parse import urlencode

import pandas as pd
import streamlit as st

from advanced_visualization.core.artifacts import available_data_sources, read_csv
from advanced_visualization.ui.styles import inject_css
from advanced_visualization.views.extra_views.registry import extra_view_options, get_extra_view


@st.cache_data(show_spinner=False)
def read_source_csv(path: str, modified_ns: int) -> pd.DataFrame:
    return read_csv(Path(path))


def _source_options() -> list[dict[str, object]]:
    sources = [source for source in available_data_sources() if Path(source["path"]).exists()]
    return sources


def _default_source_index(sources: list[dict[str, object]], selected_path: str) -> int:
    for index, source in enumerate(sources):
        if str(source["path"]) == selected_path:
            return index
    return 0


def _launch_state() -> dict:
    return {
        "view_key": st.session_state.get("extra_view_key", ""),
        "source_path": st.session_state.get("extra_view_source_path", ""),
        "source_label": st.session_state.get("extra_view_source_label", ""),
        "model_key": st.session_state.get("extra_view_model_key", ""),
        "artifact_dir": st.session_state.get("extra_view_artifact_dir", ""),
    }


def _query_value(name: str) -> str:
    value = st.query_params.get(name, "")
    if isinstance(value, list):
        return str(value[0]) if value else ""
    return str(value)


def query_workspace_state() -> dict:
    return {
        "view_key": _query_value("workspace_view"),
        "source_path": _query_value("workspace_source"),
        "source_label": _query_value("workspace_label"),
        "model_key": _query_value("workspace_model_key"),
        "artifact_dir": _query_value("workspace_artifact_dir"),
    }


def _set_launch_state(view_key: str, source: dict[str, object]) -> None:
    st.session_state["extra_view_key"] = view_key
    st.session_state["extra_view_source_path"] = str(source["path"])
    st.session_state["extra_view_source_label"] = str(source["label"])
    st.session_state["extra_view_model_key"] = str(source.get("model_key") or "")
    artifact_dir = source.get("artifact_dir")
    st.session_state["extra_view_artifact_dir"] = str(artifact_dir) if artifact_dir else ""


def _workspace_url(view_key: str, source: dict[str, object]) -> str:
    artifact_dir = source.get("artifact_dir")
    params = {
        "workspace_view": view_key,
        "workspace_source": str(source["path"]),
        "workspace_label": str(source["label"]),
        "workspace_model_key": str(source.get("model_key") or ""),
        "workspace_artifact_dir": str(artifact_dir) if artifact_dir else "",
    }
    return "?" + urlencode(params)


def _render_launch_controls() -> None:
    sources = _source_options()
    views = extra_view_options()
    if not sources:
        st.error("No configured CSV/artifact sources exist. Add a model in Settings first.")
        return

    state = _launch_state()
    source_labels = [str(source["label"]) for source in sources]
    selected_source = st.selectbox(
        "CSV / artifact source",
        source_labels,
        index=_default_source_index(sources, state["source_path"]),
    )
    source = sources[source_labels.index(selected_source)]

    model_types = sorted({view.model_type for view in views})
    selected_model_type = st.selectbox("Model type", model_types)
    model_views = [view for view in views if view.model_type == selected_model_type]

    view_labels = [view.label for view in model_views]
    selected_view_label = st.selectbox("Workspace", view_labels)
    view = model_views[view_labels.index(selected_view_label)]

    st.caption(view.description)
    st.caption(f"Source path: {source['path']}")
    if source.get("model_key"):
        st.caption(f"Model key: {source['model_key']}")

    workspace_url = _workspace_url(view.key, source)
    st.link_button("Open workspace in new tab", workspace_url, type="primary", use_container_width=True)
    st.caption("This opens an independent workspace page and keeps the current app page unchanged.")

    with st.expander("Advanced", expanded=False):
        if st.button("Open workspace in this tab", use_container_width=True):
            _set_launch_state(view.key, source)
            st.rerun()


def _render_workspace_state(state: dict) -> None:
    if not state["view_key"] or not state["source_path"]:
        st.info("Choose a source and workspace, then launch.")
        return

    path = Path(state["source_path"])
    if not path.is_file():
        st.error(f"Launched CSV no longer exists: {path}")
        return

    view = get_extra_view(state["view_key"])
    df = read_source_csv(str(path), path.stat().st_mtime_ns)
    missing = view.missing_columns(df)
    if missing:
        st.warning(
            f"{view.label} workspace can launch, but the selected CSV is missing required column(s): "
            + ", ".join(missing)
        )

    source = {
        "label": state["source_label"] or path.name,
        "path": path,
        "model_key": state["model_key"],
        "artifact_dir": Path(state["artifact_dir"]) if state["artifact_dir"] else None,
    }
    view.render(df, source, view.config)


def _render_active_view() -> None:
    _render_workspace_state(_launch_state())


def render_workspace_from_query() -> None:
    _render_workspace_state(query_workspace_state())


def has_workspace_query() -> bool:
    state = query_workspace_state()
    return bool(state["view_key"] and state["source_path"])


def main() -> None:
    inject_css()
    st.markdown(
        """
        <div class="app-hero">
          <h1>Launch Workspace</h1>
          <p>Select a model artifact and open a model-specific clone with image review, feature projection, and configured Grad-CAM layer inspection.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Launcher", expanded=not bool(st.session_state.get("extra_view_key"))):
        _render_launch_controls()

    _render_active_view()
