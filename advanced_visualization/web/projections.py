"""Feature projection service with bounded in-memory result caching."""

from __future__ import annotations

import hashlib
import json
import threading
from collections import OrderedDict

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from advanced_visualization.core.projection import ProjectionParameters, project_matrix
from advanced_visualization.web.models import ProjectionRequest


def _labels(frame: pd.DataFrame, column: str) -> pd.Series:
    if not column or column not in frame.columns:
        return pd.Series("All", index=frame.index, dtype="object")
    return frame[column].fillna("Missing").astype(str)


def _item_ids(frame: pd.DataFrame, column: str) -> pd.Series:
    if not column or column not in frame.columns:
        return pd.Series("", index=frame.index, dtype="object")
    return frame[column].fillna("").astype(str)


def _stable_sample(frame: pd.DataFrame, rows: int, random_state: int) -> pd.DataFrame:
    if len(frame) <= rows:
        return frame
    keys = pd.DataFrame(
        {
            "row_id": frame["__row_id"].to_numpy(),
            "random_state": np.full(len(frame), random_state, dtype=np.int64),
        }
    )
    ranks = pd.util.hash_pandas_object(keys, index=False).to_numpy()
    positions = np.argsort(ranks, kind="stable")[:rows]
    return frame.iloc[positions].sort_index()


def _limit_per_class(
    frame: pd.DataFrame,
    column: str,
    default_rows: int | None,
    rows_by_class: dict[str, int],
    random_state: int,
) -> pd.DataFrame:
    if default_rows is None and not rows_by_class:
        return frame
    labels = _labels(frame, column)
    sampled = []
    for label, index in labels.groupby(labels, sort=True).groups.items():
        group = frame.loc[index]
        rows = rows_by_class.get(label, default_rows)
        sampled.append(
            _stable_sample(group, rows, random_state) if rows is not None else group
        )
    return pd.concat(sampled).sort_index() if sampled else frame.iloc[0:0]


def _complete_feature_mask(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    """Find complete feature rows without converting columns that are already numeric."""
    numeric_columns = [
        column for column in columns if is_numeric_dtype(frame[column].dtype)
    ]
    non_numeric_columns = [
        column for column in columns if column not in numeric_columns
    ]
    valid = pd.Series(True, index=frame.index, dtype=bool)
    if numeric_columns:
        valid &= frame[numeric_columns].notna().all(axis=1)
    for column in non_numeric_columns:
        valid &= pd.to_numeric(frame[column], errors="coerce").notna()
    return valid


class ProjectionService:
    def __init__(self, max_entries: int = 12) -> None:
        self._max_entries = max_entries
        self._cache: OrderedDict[str, dict] = OrderedDict()
        self._lock = threading.RLock()

    def project(
        self, frame: pd.DataFrame, request: ProjectionRequest, version: str
    ) -> dict:
        columns = [
            column for column in request.feature_columns if column in frame.columns
        ]
        if not columns:
            raise ValueError("No valid feature columns were selected.")
        if request.method == "pca" and len(columns) < 2:
            raise ValueError("PCA requires at least two feature columns.")
        key_data = request.model_dump() | {"version": version}
        key_data["categorical_filters"] = {
            column: sorted(set(values))
            for column, values in request.categorical_filters.items()
        }
        key = hashlib.sha1(
            json.dumps(key_data, sort_keys=True).encode("utf-8")
        ).hexdigest()
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]

        candidate = frame
        for column, values in request.categorical_filters.items():
            if column not in candidate.columns:
                continue
            if not values:
                candidate = candidate.iloc[0:0]
                break
            candidate = candidate[
                candidate[column].fillna("Missing").astype(str).isin(values)
            ]
        if request.method == "lda":
            if (
                not request.color_column
                or request.color_column not in candidate.columns
            ):
                raise ValueError("LDA requires a valid class/group column in Color by.")
            candidate = candidate[candidate[request.color_column].notna()]
        has_class_limits = request.max_rows_per_class is not None or bool(
            request.max_rows_by_class
        )
        if has_class_limits and request.color_column not in candidate.columns:
            raise ValueError("Per-class limit requires a valid Color by column.")
        candidate = candidate.loc[_complete_feature_mask(candidate, columns)]
        available_labels = _labels(candidate, request.color_column)
        available_counts = available_labels.value_counts().to_dict()
        working = _limit_per_class(
            candidate,
            request.color_column,
            request.max_rows_per_class,
            request.max_rows_by_class,
            request.random_state,
        )
        working = _stable_sample(working, request.max_rows, request.random_state)
        matrix = working[columns].to_numpy(dtype=np.float32)
        if len(matrix) < 3:
            raise ValueError("At least three complete feature rows are required.")
        labels = (
            working[request.color_column].astype(str).to_numpy()
            if request.method == "lda"
            else None
        )
        projection = project_matrix(
            matrix,
            ProjectionParameters(
                method=request.method,
                scale=request.scale,
                perplexity=request.perplexity,
                umap_neighbors=request.umap_neighbors,
                umap_min_dist=request.umap_min_dist,
                random_state=request.random_state,
            ),
            labels=labels,
        )
        coords = projection.values

        displayed_labels = _labels(working, request.color_column)
        displayed_item_ids = _item_ids(working, request.item_id_column)
        displayed_counts = displayed_labels.value_counts().to_dict()
        points = []
        for position, (row_id, label, item_id) in enumerate(
            zip(
                working["__row_id"].tolist(),
                displayed_labels.tolist(),
                displayed_item_ids.tolist(),
            )
        ):
            points.append(
                {
                    "x": float(coords[position, 0]),
                    "y": float(coords[position, 1]),
                    "row_id": int(row_id),
                    "label": label,
                    "item_id": item_id,
                }
            )
        class_counts = [
            {
                "label": label,
                "available": int(available_counts[label]),
                "displayed": int(displayed_counts.get(label, 0)),
            }
            for label in sorted(available_counts)
        ]
        result = {
            "method": request.method,
            "subtitle": projection.note,
            "rows": len(points),
            "available_rows": len(candidate),
            "class_counts": class_counts,
            "points": points,
        }
        with self._lock:
            self._cache[key] = result
            self._cache.move_to_end(key)
            while len(self._cache) > self._max_entries:
                self._cache.popitem(last=False)
        return result


projection_service = ProjectionService()
