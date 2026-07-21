"""Reusable, UI-independent dataframe filtering primitives."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import pandas as pd


def apply_text_search(
    frame: pd.DataFrame,
    query: str,
    columns: Iterable[str],
) -> pd.DataFrame:
    """Return rows containing a literal, case-insensitive query in known columns."""
    query = query.strip()
    known_columns = [column for column in columns if column in frame.columns]
    if not query or not known_columns:
        return frame

    mask = pd.Series(False, index=frame.index)
    for column in known_columns:
        mask |= (
            frame[column]
            .astype(str)
            .str.contains(
                query,
                case=False,
                na=False,
                regex=False,
            )
        )
    return frame[mask]


def apply_categorical_filters(
    frame: pd.DataFrame,
    filters: Mapping[str, Sequence[str]],
    *,
    missing_label: str = "Missing",
) -> pd.DataFrame:
    """Apply allow-list filters, ignoring columns that are not in the frame."""
    filtered = frame
    for column, allowed in filters.items():
        if column not in filtered.columns:
            continue
        if not allowed:
            return filtered.iloc[0:0]
        filtered = filtered[
            filtered[column].fillna(missing_label).astype(str).isin(allowed)
        ]
    return filtered


def apply_numeric_ranges(
    frame: pd.DataFrame,
    ranges: Mapping[str, tuple[float, float]],
) -> pd.DataFrame:
    """Keep rows whose known numeric values fall inside inclusive ranges."""
    filtered = frame
    for column, (minimum, maximum) in ranges.items():
        if column not in filtered.columns:
            continue
        values = pd.to_numeric(filtered[column], errors="coerce")
        filtered = filtered[values.between(minimum, maximum, inclusive="both")]
    return filtered
