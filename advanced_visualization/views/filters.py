"""Dataframe enrichment and filtering for the image-review view."""

from __future__ import annotations

from typing import Optional

import pandas as pd
import streamlit as st

from advanced_visualization.core.dataframe_filters import (
    apply_categorical_filters,
    apply_numeric_ranges,
    apply_text_search,
)
from advanced_visualization.core.evaluation import (
    attach_binary_evaluation,
    evaluate_binary,
)
from advanced_visualization.core.gradcam_cache import (
    gradcam_file_index,
    gradcam_roots,
    resolve_gradcam_path,
)
from advanced_visualization.core.images import image_cache_digest


@st.cache_data(show_spinner=False)
def cached_gradcam_file_index(
    root: str, method: str = "", modified_ns: int = 0
) -> dict[str, str]:
    return gradcam_file_index(root, method=method)


def add_failure_columns(
    df: pd.DataFrame,
    truth_column: Optional[str],
    prediction_column: Optional[str],
    threshold: float,
    positive_value: int,
    negative_value: int,
    invalid_value: int,
) -> pd.DataFrame:
    return attach_binary_evaluation(
        df,
        evaluate_binary(
            df,
            truth_column,
            prediction_column,
            threshold,
            positive_value=positive_value,
            negative_value=negative_value,
            invalid_value=invalid_value,
        ),
    )


def apply_truth_rows(df: pd.DataFrame, truth_rows: str) -> pd.DataFrame:
    if truth_rows == "All rows":
        return df
    if truth_rows == "Valid only: 0 and 1":
        return df[df["__truth_is_valid"]]
    if truth_rows == "Positive only: 1":
        return df[df["__actual_positive"] & df["__truth_is_valid"]]
    if truth_rows == "Negative only: 0":
        return df[~df["__actual_positive"] & df["__truth_is_valid"]]
    if truth_rows == "Invalid only: -1":
        return df[df["__truth_is_invalid"]]
    return df


def apply_failure_mode(
    df: pd.DataFrame, mode: str, high_conf: float, low_conf: float
) -> pd.DataFrame:
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


def add_gradcam_columns(df: pd.DataFrame, controls: dict, caption=None) -> pd.DataFrame:
    if df.empty:
        enriched = df.copy()
        enriched["__gradcam_path"] = ""
        enriched["__has_gradcam"] = False
        return enriched
    enriched = df.copy()

    paths = None
    image_column = controls["image_column"]
    active_stem = controls.get("active_csv_stem")
    method = controls.get("cam_method", "")
    roots = [
        root
        for root in gradcam_roots(active_stem, controls.get("gradcam_dir", ""))
        if root.exists()
    ]
    should_preload = image_column and image_column in enriched.columns and roots
    if should_preload:
        index = {}
        loaded_count = 0
        for root in roots:
            root_index = cached_gradcam_file_index(
                str(root), method, root.stat().st_mtime_ns
            )
            loaded_count += len(root_index)
            index.update(
                {
                    digest: path
                    for digest, path in root_index.items()
                    if digest not in index
                }
            )
        digests = enriched[image_column].map(image_cache_digest)
        paths = digests.map(lambda digest: index.get(digest, "") if digest else "")
        if caption:
            caption(
                f"Preloaded {loaded_count:,} Grad-CAM files from {len(roots)} folder(s)"
            )

    if paths is None:
        paths = df.apply(lambda row: resolve_gradcam_path(row, controls), axis=1).map(
            lambda path: str(path) if path is not None else ""
        )
    elif controls["gradcam_column"] or controls["gradcam_dir"]:
        missing = paths.eq("")
        if missing.any():
            fallback = (
                enriched[missing]
                .apply(lambda row: resolve_gradcam_path(row, controls), axis=1)
                .map(lambda path: str(path) if path is not None else "")
            )
            paths.loc[missing] = fallback

    enriched["__gradcam_path"] = paths.map(
        lambda path: str(path) if path is not None else ""
    )
    enriched["__has_gradcam"] = enriched["__gradcam_path"].ne("")
    return enriched


def apply_all_filters(
    df: pd.DataFrame, controls: dict, caption=None, include_gradcam: bool = True
) -> pd.DataFrame:
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
    filtered = apply_failure_mode(
        filtered, controls["mode"], controls["high_conf"], controls["low_conf"]
    )
    filtered = apply_text_search(
        filtered, controls["text_query"], controls["search_columns"]
    )
    filtered = apply_categorical_filters(filtered, controls["categorical_filters"])
    filtered = apply_numeric_ranges(filtered, controls["numeric_ranges"])
    if include_gradcam or controls["only_prepared_gradcam"]:
        filtered = add_gradcam_columns(filtered, controls, caption=caption)
        if controls["only_prepared_gradcam"]:
            filtered = filtered[filtered["__has_gradcam"]]

    sort_by = controls["sort_by"]
    if sort_by == "confidence desc":
        filtered = filtered.sort_values(
            "__confidence", ascending=False, na_position="last"
        )
    elif sort_by == "confidence asc":
        filtered = filtered.sort_values(
            "__confidence", ascending=True, na_position="last"
        )
    elif sort_by == "prediction desc":
        filtered = filtered.sort_values(
            "__prediction_score", ascending=False, na_position="last"
        )
    elif sort_by == "prediction asc":
        filtered = filtered.sort_values(
            "__prediction_score", ascending=True, na_position="last"
        )
    return filtered
