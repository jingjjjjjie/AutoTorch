"""Configurable layered Grad-CAM workspace section."""
from __future__ import annotations

import html
from pathlib import Path
from typing import Iterable

import pandas as pd
import streamlit as st

from advanced_visualization.core.columns import categorical_columns, first_existing
from advanced_visualization.core.config import ID_COLUMNS, SUBCLASS_COLUMNS, all_model_runs, gradcam_artifact_root
from advanced_visualization.core.gradcam_cache import gradcam_file_index
from advanced_visualization.core.images import image_cache_digests, valid_image
from advanced_visualization.core.settings import load_settings
from advanced_visualization.ui.components import (
    format_card_value,
    preview_height,
    render_bottomless_controls,
    render_load_more,
    render_pager,
)
from advanced_visualization.ui.zoom import render_zoomable_images
from advanced_visualization.views.filters import (
    add_failure_columns,
    apply_categorical_filters,
    apply_failure_mode,
    apply_text_search,
    apply_truth_rows,
)


def _option_index(options: list | tuple, value, fallback: int = 0) -> int:
    return options.index(value) if value in options else fallback


def _first_existing(columns: Iterable[str], candidates: Iterable[str]) -> str:
    column_list = [str(column) for column in columns]
    lowered = {column.lower(): column for column in column_list}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return ""


def _path_exists(value) -> bool:
    return valid_image(value) is not None


def _column_has_values(df: pd.DataFrame, column: str) -> bool:
    if column not in df.columns:
        return False
    values = df[column].dropna().astype(str).str.strip()
    return bool(values.ne("").any())


@st.cache_data(show_spinner=False)
def _cached_gradcam_index(root: str, method: str, modified_ns: int) -> dict[str, str]:
    del modified_ns
    return gradcam_file_index(root, method=method)


def _branches(config: dict) -> list[dict]:
    return [branch for branch in config.get("branches", []) if branch.get("key")]


def _layers(config: dict) -> list[dict]:
    return [layer for layer in config.get("layers", []) if layer.get("key")]


def _branch_label(branch: dict) -> str:
    return str(branch.get("label") or branch.get("key"))


def _layer_label(layer: dict) -> str:
    return str(layer.get("label") or layer.get("key"))


def _column_from_template(template: str, branch: dict, layer: dict, config: dict) -> str:
    return template.format(
        branch=branch.get("key", ""),
        layer=layer.get("key", ""),
        score=config.get("score", ""),
        model_type=config.get("model_type", ""),
    )


def gradcam_column(branch: dict, layer: dict, config: dict) -> str:
    candidates = layer.get("column_candidates") or []
    if candidates:
        return str(candidates[0])
    template = str(layer.get("column_template") or config.get("column_template") or "{branch}_{layer}_gradcam_path")
    return _column_from_template(template, branch, layer, config)


def gradcam_candidates(branch: dict, layer: dict, config: dict) -> list[str]:
    candidates = [str(column) for column in layer.get("column_candidates", []) if str(column).strip()]
    column = gradcam_column(branch, layer, config)
    if column and column not in candidates:
        candidates.insert(0, column)
    return candidates


def first_existing_gradcam_column(df: pd.DataFrame, branch: dict, layer: dict, config: dict) -> str:
    for column in gradcam_candidates(branch, layer, config):
        if column in df.columns:
            return column
    return gradcam_column(branch, layer, config)


def _selected_branches(label: str, branches: list[dict]) -> list[dict]:
    if label == "All branches":
        return branches
    for branch in branches:
        if _branch_label(branch) == label:
            return [branch]
    return branches[:1]


def _available_layers(df: pd.DataFrame, config: dict) -> list[dict]:
    available = []
    for layer in _layers(config):
        for branch in _branches(config):
            if any(_column_has_values(df, column) for column in gradcam_candidates(branch, layer, config)):
                available.append(layer)
                break
    return available or _layers(config)


def _branch_image_default(columns: list[str], branch: dict) -> str:
    return _first_existing(columns, branch.get("image_candidates", []))


def _branch_image_key(branch: dict) -> str:
    return f"{branch.get('key')}_image_column"


def _source_prediction_column(source: dict, columns: list[str]) -> str:
    model_key = str(source.get("model_key") or "")
    model = all_model_runs().get(model_key) if model_key else None
    if model and model.prediction_column in columns:
        return model.prediction_column
    for configured_model in load_settings().models:
        if configured_model.key == model_key and configured_model.prediction_column in columns:
            return configured_model.prediction_column
    if "__prediction_column" in columns:
        return "__prediction_column"
    return ""


def _gradcam_method(layer: dict, config: dict) -> str:
    return str(layer.get("method") or config.get("method") or "gradcam")


def _prepared_gradcam_roots(source: dict) -> list[Path]:
    roots: list[Path] = []
    artifact_dir = source.get("artifact_dir")
    if artifact_dir:
        roots.append(Path(artifact_dir).expanduser() / "gradcam")

    model_key = str(source.get("model_key") or "")
    if model_key:
        root = gradcam_artifact_root(model_key)
        if root is not None:
            roots.append(root)

    unique_roots = []
    seen = set()
    for root in roots:
        key = str(root)
        if key not in seen:
            unique_roots.append(root)
            seen.add(key)
    return unique_roots


def _has_prepared_gradcam_fallback(source: dict, layer: dict, config: dict) -> bool:
    method = _gradcam_method(layer, config)
    for root in _prepared_gradcam_roots(source):
        if root.exists() and _cached_gradcam_index(str(root), method, root.stat().st_mtime_ns):
            return True
    return False


def _prepared_gradcam_path(row: pd.Series, controls: dict, branch: dict) -> Path | None:
    image_column = controls["image_columns"].get(branch["key"])
    if not image_column or image_column not in row.index:
        return None

    method = _gradcam_method(controls["layer"], controls["config"])
    for root in _prepared_gradcam_roots(controls["source"]):
        if not root.exists():
            continue
        index = _cached_gradcam_index(str(root), method, root.stat().st_mtime_ns)
        for digest in image_cache_digests(row[image_column]):
            gradcam_path = index.get(digest)
            if gradcam_path:
                return Path(gradcam_path)
    return None


def resolved_gradcam_path(row: pd.Series, controls: dict, branch: dict) -> Path | str | None:
    gradcam_column = first_existing_gradcam_column(pd.DataFrame([row]), branch, controls["layer"], controls["config"])
    if gradcam_column in row.index:
        path = valid_image(row[gradcam_column])
        if path is not None:
            return path
    return _prepared_gradcam_path(row, controls, branch)


def sidebar_controls(df: pd.DataFrame, source: dict, config: dict) -> dict:
    settings = load_settings()
    review = settings.review
    columns = df.columns.astype(str).tolist()
    cats = categorical_columns(df)
    branches = _branches(config)
    layers = _available_layers(df, config)

    id_default = first_existing(columns, ID_COLUMNS) or columns[0]
    subclass_default = first_existing(columns, SUBCLASS_COLUMNS) or _first_existing(columns, config.get("metadata_columns", []))
    truth_default = "label" if "label" in columns else subclass_default
    pred_default = _source_prediction_column(source, columns) or _first_existing(columns, config.get("prediction_candidates", []))
    branch_labels = [_branch_label(branch) for branch in branches]
    target_options = branch_labels + (["All branches"] if len(branches) > 1 else [])
    default_layer_key = str(config.get("default_layer") or "")
    default_layer_label = next((_layer_label(layer) for layer in layers if layer.get("key") == default_layer_key), _layer_label(layers[0]) if layers else "")
    layer_labels = [_layer_label(layer) for layer in layers]

    with st.sidebar.form(f"{config.get('model_type', 'extra')}_layered_gradcam_form"):
        submitted = st.form_submit_button("Update workspace", type="primary", use_container_width=True)

        st.header(str(config.get("label") or "Layered Grad-CAM"))
        target = st.radio("Target", target_options, index=len(target_options) - 1 if "All branches" in target_options else 0)
        layer_label = st.radio("Layer", layer_labels, index=_option_index(layer_labels, default_layer_label))
        only_prepared = st.checkbox("Only rows with selected artifact", value=True)

        st.header("Columns")
        item_id_column = st.selectbox("Item ID", columns, index=columns.index(id_default))
        truth_column = st.selectbox("Truth column", ["None"] + columns, index=(columns.index(truth_default) + 1 if truth_default in columns else 0))
        prediction_column = st.selectbox("Prediction score", ["None"] + columns, index=(columns.index(pred_default) + 1 if pred_default in columns else 0))
        subclass_column = st.selectbox("Subclass/group", ["None"] + cats, index=(cats.index(subclass_default) + 1 if subclass_default in cats else 0))
        image_columns = {}
        for branch in branches:
            default = _branch_image_default(columns, branch)
            image_columns[branch["key"]] = st.selectbox(
                f"{_branch_label(branch)} image",
                ["None"] + columns,
                index=(columns.index(default) + 1 if default in columns else 0),
            )

        st.header("Failure Logic")
        prediction_threshold = st.slider("Prediction threshold", 0.0, 1.0, float(review.get("prediction_threshold", 0.5)), 0.01)
        truth_row_options = ["Valid only: 0 and 1", "Positive only: 1", "Negative only: 0", "Invalid only: -1", "All rows"]
        truth_rows = st.radio("Truth rows", truth_row_options, index=_option_index(truth_row_options, review.get("default_truth_rows", "Valid only: 0 and 1")))
        failure_view_options = [
            "All rows",
            "Failures only",
            "High-confidence failures",
            "Low-confidence failures",
            "False positives",
            "False negatives",
            "Correct only",
        ]
        failure_view = st.radio("Failure view", failure_view_options, index=_option_index(failure_view_options, review.get("default_failure_view", "All rows")))
        high_conf = st.slider("High confidence >=", 0.0, 1.0, float(review.get("high_confidence", 0.9)), 0.01)
        low_conf = st.slider("Low confidence <=", 0.0, 1.0, float(review.get("low_confidence", 0.6)), 0.01)

        st.header("Filters")
        search_columns = st.multiselect("Search columns", columns, default=[column for column in (item_id_column, subclass_column) if column and column != "None"])
        text_query = st.text_input("Search text")
        default_filter_columns = [column for column in config.get("metadata_columns", []) if column in cats]
        filter_columns = st.multiselect("Categorical filters", cats, default=default_filter_columns)
        categorical_filters = {}
        for column in filter_columns:
            values = sorted(df[column].fillna("Missing").astype(str).unique().tolist())
            categorical_filters[column] = st.multiselect(column, values, default=values)

        st.header("Layout")
        browse_mode = st.radio("Browse mode", ["Bottomless scroll", "Pages"], index=0, horizontal=True)
        page_size = st.select_slider("Page size", options=[12, 24, 48, 96], value=min(48, int(review.get("page_size", 48))))
        columns_per_row = st.slider("Cards per row", 2, 8, 6)
        show_card_metadata = st.checkbox("Show card metadata", value=False)
        sort_by = st.selectbox("Sort", ["confidence desc", "confidence asc", "prediction desc", "prediction asc", "row order"], index=0)

    if submitted:
        st.session_state["layered_gradcam_visible_count"] = page_size
        st.session_state["layered_gradcam_page"] = 1

    selected_layer = next(layer for layer in layers if _layer_label(layer) == layer_label)
    return {
        "source": source,
        "config": config,
        "target": target,
        "branches": _selected_branches(target, branches),
        "all_branches": branches,
        "layer": selected_layer,
        "only_prepared": only_prepared,
        "item_id_column": item_id_column,
        "truth_column": None if truth_column == "None" else truth_column,
        "prediction_column": None if prediction_column == "None" else prediction_column,
        "subclass_column": None if subclass_column == "None" else subclass_column,
        "image_columns": {key: (None if value == "None" else value) for key, value in image_columns.items()},
        "prediction_threshold": prediction_threshold,
        "truth_rows": truth_rows,
        "positive_truth_value": int(review.get("positive_truth_value", 1)),
        "negative_truth_value": int(review.get("negative_truth_value", 0)),
        "invalid_truth_value": int(review.get("invalid_truth_value", -1)),
        "high_conf": high_conf,
        "low_conf": low_conf,
        "failure_view": failure_view,
        "search_columns": search_columns,
        "text_query": text_query,
        "categorical_filters": categorical_filters,
        "browse_mode": browse_mode,
        "page_size": page_size,
        "columns_per_row": columns_per_row,
        "show_card_metadata": show_card_metadata,
        "sort_by": sort_by,
    }


def apply_filters(df: pd.DataFrame, controls: dict) -> pd.DataFrame:
    filtered = add_failure_columns(
        df,
        controls["truth_column"],
        controls["prediction_column"],
        controls["prediction_threshold"],
        controls["positive_truth_value"],
        controls["negative_truth_value"],
        controls["invalid_truth_value"],
    )
    filtered = apply_truth_rows(filtered, controls["truth_rows"])
    filtered = apply_failure_mode(filtered, controls["failure_view"], controls["high_conf"], controls["low_conf"])
    filtered = apply_text_search(filtered, controls["text_query"], controls["search_columns"])
    filtered = apply_categorical_filters(filtered, controls["categorical_filters"])

    if controls["only_prepared"]:
        mask = pd.Series(False, index=filtered.index)
        for branch in controls["branches"]:
            column = first_existing_gradcam_column(filtered, branch, controls["layer"], controls["config"])
            if column in filtered.columns:
                mask |= filtered[column].map(_path_exists)
            else:
                mask |= filtered.apply(lambda row: resolved_gradcam_path(row, controls, branch) is not None, axis=1)
        filtered = filtered[mask]

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
    available = 0
    for branch in controls["branches"]:
        column = first_existing_gradcam_column(filtered, branch, controls["layer"], controls["config"])
        if column in filtered.columns:
            available += int(filtered[column].map(_path_exists).sum())
        else:
            available += int(filtered.apply(lambda row: resolved_gradcam_path(row, controls, branch) is not None, axis=1).sum())

    cols = st.columns(5)
    cols[0].metric("Rows shown", f"{len(filtered):,}", delta=f"from {len(source):,}")
    cols[1].metric("Scored rows", f"{len(scored):,}")
    cols[2].metric("Failures", f"{failures:,}", delta=f"{failure_rate:.1%}")
    cols[3].metric("Layer", _layer_label(controls["layer"]))
    cols[4].metric("Prepared artifacts", f"{available:,}")


def _metadata_line(row: pd.Series, controls: dict) -> str:
    fields = []
    for column in [controls["subclass_column"], *controls["config"].get("metadata_columns", [])]:
        if column and column in row.index:
            value = f"{column}={format_card_value(row[column])}"
            if value not in fields:
                fields.append(value)
    for column in controls["config"].get("prediction_candidates", [])[:3]:
        if column and column in row.index:
            fields.append(f"{column}={format_card_value(row[column])}")
    return " | ".join(fields)


def _render_branch(
    row: pd.Series,
    controls: dict,
    branch: dict,
    *,
    status_label: str = "",
    status_kind: str = "",
    index_label: str = "",
) -> None:
    image_column = controls["image_columns"].get(branch["key"])
    original_path = row[image_column] if image_column and image_column in row.index else None
    gradcam_path = resolved_gradcam_path(row, controls, branch)
    label = f"{_branch_label(branch)} {_layer_label(controls['layer'])}"

    if controls["layer"].get("display") == "single":
        if not render_zoomable_images(
            [(label, gradcam_path)],
            preview_height=preview_height(controls, 1),
            status_label=status_label,
            status_kind=status_kind,
            index_label=index_label,
        ):
            st.caption(f"No {label}")
        return

    if not render_zoomable_images(
        [(f"{_branch_label(branch)} image", original_path), (label, gradcam_path)],
        preview_height=preview_height(controls, 2),
        status_label=status_label,
        status_kind=status_kind,
        index_label=index_label,
    ):
        st.caption(f"No {label}")


def render_card(row: pd.Series, controls: dict, display_index: int) -> None:
    is_failure = bool(row.get("__is_failure", False))
    pill_text = str(row.get("__failure_type", "unscored"))
    status_kind = "fail" if is_failure else "pass"
    index_label = f"#{display_index}"
    item = html.escape(str(row.get(controls["item_id_column"], row.name)))

    branches = controls["branches"]
    if len(branches) == 1:
        _render_branch(row, controls, branches[0], status_label=pill_text, status_kind=status_kind, index_label=index_label)
    else:
        columns = st.columns(len(branches))
        for column, branch in zip(columns, branches):
            with column:
                _render_branch(row, controls, branch, status_label=pill_text, status_kind=status_kind, index_label=index_label)

    if controls.get("show_card_metadata", False):
        st.markdown(f'<div class="viewer-caption">{item}</div>', unsafe_allow_html=True)
        metadata = _metadata_line(row, controls)
        if metadata:
            st.caption(metadata)
    else:
        st.markdown(f'<div class="viewer-caption compact">{item}</div>', unsafe_allow_html=True)


def render_grid(df: pd.DataFrame, controls: dict, start_index: int = 1) -> None:
    rows = list(enumerate(df.iterrows(), start=start_index))
    for offset in range(0, len(rows), controls["columns_per_row"]):
        columns = st.columns(controls["columns_per_row"], gap="small")
        for column, (display_index, (_index, row)) in zip(columns, rows[offset : offset + controls["columns_per_row"]]):
            with column:
                with st.container(border=False):
                    render_card(row, controls, display_index)


def render(df: pd.DataFrame, source: dict, config: dict) -> None:
    title = html.escape(str(config.get("label") or "Layered Grad-CAM Review"))
    description = html.escape(str(config.get("description") or "Model-specific layered Grad-CAM artifact review."))
    st.markdown(
        f"""
        <div class="app-hero">
          <h1>{title}</h1>
          <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    controls = sidebar_controls(df, source, config)
    missing = [
        first_existing_gradcam_column(df, branch, controls["layer"], config)
        for branch in controls["branches"]
        if first_existing_gradcam_column(df, branch, controls["layer"], config) not in df.columns
        and not _has_prepared_gradcam_fallback(source, controls["layer"], config)
    ]
    if missing:
        st.warning(f"Missing selected artifact column(s): {', '.join(missing)}")

    filtered = apply_filters(df, controls)
    render_summary(df, filtered, controls)

    if filtered.empty:
        st.warning("No rows match the current workspace filters.")
        return

    if controls["browse_mode"] == "Bottomless scroll":
        visible = render_bottomless_controls(len(filtered), controls["page_size"])
        render_grid(filtered.iloc[:visible], controls, start_index=1)
        render_load_more(len(filtered), controls["page_size"])
    else:
        start, end = render_pager(len(filtered), controls["page_size"])
        render_grid(filtered.iloc[start:end], controls, start_index=start + 1)
