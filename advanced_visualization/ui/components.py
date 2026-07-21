"""Reusable Streamlit components for the image-review view."""
from __future__ import annotations

import html
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

from advanced_visualization.core.gradcam_cache import resolve_gradcam_path
from advanced_visualization.core.images import valid_image
from advanced_visualization.ui.zoom import render_zoomable_images

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

def gradcam_for_row(row: pd.Series, controls: dict) -> tuple[Optional[Path], Optional[str]]:
    existing = row.get("__gradcam_path") or resolve_gradcam_path(row, controls)
    if existing:
        return Path(str(existing)), None
    return None, None

def render_summary(source: pd.DataFrame, filtered: pd.DataFrame, controls: dict) -> None:
    scored = filtered[filtered["__has_eval"]]
    failures = int(filtered["__is_failure"].sum()) if "__is_failure" in filtered else 0
    failure_rate = failures / len(scored) if len(scored) else 0.0

    prepared_gradcam = int(filtered["__has_gradcam"].sum()) if "__has_gradcam" in filtered else None

    metric_cols = st.columns(6)
    metric_cols[0].metric("Rows shown", f"{len(filtered):,}", delta=f"from {len(source):,}")
    metric_cols[1].metric("Scored rows", f"{len(scored):,}")
    metric_cols[2].metric("Failures", f"{failures:,}", delta=f"{failure_rate:.1%}")
    metric_cols[3].metric("High-conf failures", f"{int((filtered['__is_failure'] & filtered['__confidence'].ge(controls['high_conf'])).sum()):,}")
    metric_cols[4].metric("Low-conf failures", f"{int((filtered['__is_failure'] & filtered['__confidence'].le(controls['low_conf'])).sum()):,}")
    metric_cols[5].metric("Prepared Grad-CAM", f"{prepared_gradcam:,}" if prepared_gradcam is not None else "visible")

def page_bounds(total: int, page_size: int, state_namespace: str = "advanced_visualization") -> tuple[int, int, int, int]:
    total_pages = max(1, int(np.ceil(total / page_size)))
    page_key = f"{state_namespace}_page"
    current = int(st.session_state.get(page_key, 1))
    current = min(max(1, current), total_pages)
    st.session_state[page_key] = current
    start = (current - 1) * page_size
    end = min(start + page_size, total)
    return current, total_pages, start, end

def render_pager(total: int, page_size: int, state_namespace: str = "advanced_visualization") -> tuple[int, int]:
    current, total_pages, start, end = page_bounds(total, page_size, state_namespace)
    left, middle, right = st.columns([0.18, 0.64, 0.18])
    with left:
        if st.button("Previous", disabled=current <= 1, use_container_width=True, key=f"{state_namespace}_previous"):
            st.session_state[f"{state_namespace}_page"] = current - 1
            st.rerun()
    with middle:
        selected = st.number_input("Page", min_value=1, max_value=total_pages, value=current, step=1, key=f"{state_namespace}_page_input")
        if selected != current:
            st.session_state[f"{state_namespace}_page"] = int(selected)
            st.rerun()
        st.caption(f"Showing {start + 1 if total else 0:,}-{end:,} of {total:,}")
    with right:
        if st.button("Next", disabled=current >= total_pages, use_container_width=True, key=f"{state_namespace}_next"):
            st.session_state[f"{state_namespace}_page"] = current + 1
            st.rerun()
    return start, end

def visible_count(total: int, batch_size: int, state_namespace: str = "advanced_visualization") -> int:
    key = f"{state_namespace}_visible_count"
    version_key = f"{state_namespace}_loader_version"
    loader_version = "stable_manual_20260710"
    if st.session_state.get(version_key) != loader_version:
        st.session_state[version_key] = loader_version
        st.session_state[key] = batch_size
    current = int(st.session_state.get(key, batch_size))
    if current > batch_size * 6:
        current = batch_size
    current = min(max(batch_size, current), max(total, batch_size))
    st.session_state[key] = current
    return min(current, total)

def render_bottomless_controls(total: int, batch_size: int, state_namespace: str = "advanced_visualization") -> int:
    count = visible_count(total, batch_size, state_namespace)
    st.caption(f"Showing 1-{count:,} of {total:,}")
    return count

def render_load_more(total: int, batch_size: int, state_namespace: str = "advanced_visualization") -> None:
    count = visible_count(total, batch_size, state_namespace)
    if count >= total:
        st.caption(f"All {total:,} rows shown.")
        return

    left, middle, right = st.columns([0.25, 0.50, 0.25])
    with middle:
        if st.button("Load more", use_container_width=True, key=f"{state_namespace}_load_more"):
            st.session_state[f"{state_namespace}_visible_count"] = min(total, count + batch_size)
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


def compact_row_label(row: pd.Series, controls: dict) -> str:
    fields = []
    if "Data_Identity" in row.index:
        value = row.get("Data_Identity")
        if pd.notna(value) and str(value):
            fields.append(str(value))
    subclass = row.get(controls["subclass_column"], "") if controls["subclass_column"] else ""
    if pd.notna(subclass) and str(subclass):
        fields.append(str(subclass))
    if "Quality_Issue" in row.index:
        value = row.get("Quality_Issue")
        if pd.notna(value) and str(value):
            fields.append(f"Q={format_card_value(value)}")
    score = row.get("__prediction_score", np.nan)
    if pd.notna(score):
        fields.append(f"pred={float(score):.4f}")
    if not fields:
        fields.append(str(row.get(controls["item_id_column"], row.name)))
    return " | ".join(fields)


def compact_row_label_html(row: pd.Series, controls: dict) -> str:
    text = compact_row_label(row, controls)
    primary, separator, secondary = text.partition(" | ")
    if not separator:
        return html.escape(primary)
    return (
        f'<span class="caption-primary">{html.escape(primary)}</span>'
        f'<span class="caption-secondary">{html.escape(secondary)}</span>'
    )


def preview_height(controls: dict, pane_count: int) -> int:
    columns = int(controls.get("columns_per_row", 6))
    if pane_count > 1:
        return {2: 330, 3: 305, 4: 285, 5: 265, 6: 245, 7: 225, 8: 215, 9: 205, 10: 195}.get(columns, 245)
    return {2: 540, 3: 470, 4: 400, 5: 355, 6: 310, 7: 285, 8: 260, 9: 240, 10: 225}.get(columns, 310)


def render_image_cell(
    row: pd.Series,
    controls: dict,
    *,
    display_index: Optional[int] = None,
) -> None:
    image_column = controls["image_column"]
    original_path = row[image_column] if image_column else None
    view_mode = controls["view_mode"]
    gradcam_error = None
    if view_mode == "Original":
        gradcam_path = row.get("__gradcam_path") or resolve_gradcam_path(row, controls)
    else:
        gradcam_path, gradcam_error = gradcam_for_row(row, controls)

    is_failure = bool(row.get("__is_failure", False))
    pill_text = row.get("__failure_type", "unscored")
    index_label = f"#{display_index}" if display_index is not None else ""
    status_kind = "fail" if is_failure else "pass"

    if view_mode == "Original":
        if not render_zoomable_images(
            [("Original", original_path)],
            preview_height=preview_height(controls, 1),
            status_label=str(pill_text),
            status_kind=status_kind,
            index_label=index_label,
        ):
            st.caption("No image")
    elif view_mode == "Grad-CAM":
        if not render_zoomable_images(
            [("Grad-CAM", gradcam_path)],
            preview_height=preview_height(controls, 1),
            status_label=str(pill_text),
            status_kind=status_kind,
            index_label=index_label,
        ):
            st.caption("No Grad-CAM")
            if gradcam_error:
                st.caption(gradcam_error)
    else:
        if not render_zoomable_images(
            [("Original", original_path), ("Grad-CAM", gradcam_path)],
            preview_height=preview_height(controls, 2),
            status_label=str(pill_text),
            status_kind=status_kind,
            index_label=index_label,
        ):
            st.caption("Missing")
        if valid_image(original_path) is None:
            st.caption("Original missing")
        if valid_image(gradcam_path) is None:
            st.caption("Grad-CAM missing")
            if gradcam_error:
                st.caption(gradcam_error)

    if controls.get("show_card_metadata", False):
        render_filter_tags(row, controls)
        st.markdown(f'<div class="viewer-caption">{row_label(row, controls)}</div>', unsafe_allow_html=True)
        if gradcam_path:
            st.caption(Path(str(gradcam_path)).name)
    else:
        st.markdown(
            f'<div class="viewer-caption compact">{compact_row_label_html(row, controls)}</div>',
            unsafe_allow_html=True,
        )

def render_grid(page_df: pd.DataFrame, controls: dict, start_index: int = 1) -> None:
    cols_per_row = controls["columns_per_row"]
    rows = list(enumerate(page_df.iterrows(), start=start_index))
    for offset in range(0, len(rows), cols_per_row):
        columns = st.columns(cols_per_row, gap="small")
        for column, (display_index, (_index, row)) in zip(columns, rows[offset : offset + cols_per_row]):
            with column:
                with st.container(border=False):
                    render_image_cell(row, controls, display_index=display_index)

def render_breakdowns(filtered: pd.DataFrame, controls: dict) -> None:
    with st.expander("Breakdowns", expanded=False):
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
