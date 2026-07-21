"""Pure dataframe helpers for feature-space exploration."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd


FEATURE_PREFIXES = ("feature_", "feat_", "embedding_", "emb_")
FEATURE_PATTERN = re.compile(
    r"(?:^|_)(feature|feat|embedding|emb)[_-]?\d+$", re.IGNORECASE
)
PREDICTION_PATTERN = re.compile(r"(pred|prob|score)", re.IGNORECASE)
PRED_BUCKET_COLUMN = "__model_pred_bucket"
PRED_DISPLAY_COLUMN = "__model_pred"


def feature_columns(frame: pd.DataFrame) -> list[str]:
    numeric = frame.select_dtypes(include=[np.number]).columns.tolist()
    matches = [column for column in numeric if FEATURE_PATTERN.search(str(column))]
    if not matches:
        prefixes = ", ".join(FEATURE_PREFIXES)
        raise ValueError(
            f"No numeric feature columns found. Expected columns prefixed with: {prefixes}"
        )
    return matches


def metadata_columns(
    frame: pd.DataFrame,
    features: list[str],
    candidates: list[str],
) -> list[str]:
    excluded = set(features)
    result = []
    for column in candidates:
        if column in excluded or column not in frame.columns:
            continue
        unique_count = frame[column].nunique(dropna=True)
        if frame[column].dtype == "object" or unique_count <= min(
            80, max(12, len(frame) // 5)
        ):
            result.append(column)
    return result


def parse_merge_mapping(raw_mapping: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for line_number, raw_line in enumerate(raw_mapping.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        if "=" not in line:
            raise ValueError(f"Merge rule line {line_number} is missing '='.")
        merged_name, values = line.split("=", 1)
        merged_name = merged_name.strip()
        if not merged_name:
            raise ValueError(
                f"Merge rule line {line_number} has an empty merged class name."
            )
        source_values = [value.strip() for value in values.split(",") if value.strip()]
        if not source_values:
            raise ValueError(f"Merge rule line {line_number} has no source values.")
        mapping.update({value: merged_name for value in source_values})
    return mapping


def apply_merge_mapping(series: pd.Series, raw_mapping: str) -> pd.Series:
    if not raw_mapping.strip():
        return series.astype(str)
    mapping = parse_merge_mapping(raw_mapping)
    return series.astype(str).map(lambda value: mapping.get(value, value))


def limit_named_group(
    frame: pd.DataFrame,
    *,
    group_column: str,
    group_name: str,
    max_rows: int,
    random_state: int,
) -> pd.DataFrame:
    if group_column not in frame.columns:
        return frame
    group_mask = (
        frame[group_column].astype(str).str.casefold().eq(group_name.casefold())
    )
    group = frame[group_mask]
    if len(group) <= max_rows:
        return frame
    sampled = group.sample(n=max_rows, random_state=random_state)
    limited = pd.concat([sampled, frame[~group_mask]], axis=0).sort_index()
    limited.attrs["sampling_note"] = (
        f"Random-sampled {group_name} from {len(group):,} to {max_rows:,} rows."
    )
    return limited


def limit_rows(frame: pd.DataFrame, max_rows: int, random_state: int) -> pd.DataFrame:
    if max_rows <= 0 or len(frame) <= max_rows:
        return frame
    sampled = frame.sample(n=max_rows, random_state=random_state).sort_index()
    sampled.attrs.update(frame.attrs)
    sampled.attrs["plot_sampling_note"] = (
        f"Random-sampled plot rows from {len(frame):,} to {max_rows:,}."
    )
    return sampled


def add_prediction_columns(
    frame: pd.DataFrame,
    prediction_column: str | None,
    threshold: float,
) -> pd.DataFrame:
    if not prediction_column or prediction_column not in frame.columns:
        return frame
    result = frame.copy()
    prediction = pd.to_numeric(result[prediction_column], errors="coerce")
    result[PRED_DISPLAY_COLUMN] = prediction
    result[PRED_BUCKET_COLUMN] = np.select(
        [prediction.isna(), prediction.ge(threshold)],
        ["missing pred", f"pred >= {threshold:.2f}"],
        default=f"pred < {threshold:.2f}",
    )
    return result
