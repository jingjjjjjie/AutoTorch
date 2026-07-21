"""Column inference and lightweight CSV schema helpers."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Optional

import numpy as np
import pandas as pd

from advanced_visualization.core.config import (
    ID_COLUMNS,
    IMAGE_COLUMNS,
    SUBCLASS_COLUMNS,
)


PREDICTION_PATTERN = re.compile(r"(pred|prob|score|result)", re.IGNORECASE)
GRADCAM_PATTERN = re.compile(r"(grad.?cam|cam|heatmap|overlay)", re.IGNORECASE)


def first_existing(columns: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    lowered = {str(column).lower(): str(column) for column in columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def image_path_columns(df: pd.DataFrame) -> list[str]:
    columns = []
    for column in df.columns:
        lower = str(column).lower()
        if (
            column in IMAGE_COLUMNS
            or "path" in lower
            or GRADCAM_PATTERN.search(str(column))
        ):
            columns.append(column)
    return columns


def prediction_columns(df: pd.DataFrame) -> list[str]:
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    return [
        column for column in numeric_columns if PREDICTION_PATTERN.search(str(column))
    ]


def categorical_columns(df: pd.DataFrame) -> list[str]:
    columns = []
    for column in df.columns:
        unique_count = df[column].nunique(dropna=True)
        if df[column].dtype == "object" or unique_count <= min(
            120, max(12, len(df) // 8)
        ):
            columns.append(column)
    return columns


def numeric_filter_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=[np.number]).columns.tolist()


def infer_standard_columns(
    df: pd.DataFrame, prediction_column: str = ""
) -> dict[str, str]:
    columns = df.columns.astype(str).tolist()
    image_columns = image_path_columns(df)
    prediction_candidates = prediction_columns(df)
    subclass_default = first_existing(columns, SUBCLASS_COLUMNS)
    return {
        "item_id_column": first_existing(columns, ID_COLUMNS) or columns[0],
        "image_column": first_existing(columns, IMAGE_COLUMNS)
        or (image_columns[0] if image_columns else ""),
        "subclass_column": subclass_default or "",
        "truth_column": "label" if "label" in df.columns else (subclass_default or ""),
        "prediction_column": (
            prediction_column
            if prediction_column in df.columns
            else (prediction_candidates[-1] if prediction_candidates else "")
        ),
    }
