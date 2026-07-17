"""Feature projection service with bounded in-memory result caching."""
from __future__ import annotations

import hashlib
import json
import os
import threading
from collections import OrderedDict

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/autotorch_visualization_numba")

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from advanced_visualization.web.models import ProjectionRequest


MAX_TSNE_ROWS = 5000
MAX_UMAP_ROWS = 50000


def _labels(frame: pd.DataFrame, column: str) -> pd.Series:
    if not column or column not in frame.columns:
        return pd.Series("All", index=frame.index, dtype="object")
    return frame[column].fillna("Missing").astype(str)


def _stable_sample(frame: pd.DataFrame, rows: int, random_state: int) -> pd.DataFrame:
    if len(frame) <= rows:
        return frame
    keys = pd.DataFrame({
        "row_id": frame["__row_id"].to_numpy(),
        "random_state": np.full(len(frame), random_state, dtype=np.int64),
    })
    ranks = pd.util.hash_pandas_object(keys, index=False).to_numpy()
    positions = np.argsort(ranks, kind="stable")[:rows]
    return frame.iloc[positions].sort_index()


def _limit_per_class(
    frame: pd.DataFrame, column: str, rows: int | None, random_state: int
) -> pd.DataFrame:
    if rows is None:
        return frame
    labels = _labels(frame, column)
    group_indices = labels.groupby(labels, sort=True).groups.values()
    sampled = [_stable_sample(frame.loc[index], rows, random_state) for index in group_indices]
    return pd.concat(sampled).sort_index() if sampled else frame.iloc[0:0]


def _complete_feature_mask(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    """Find complete feature rows without converting columns that are already numeric."""
    numeric_columns = [column for column in columns if is_numeric_dtype(frame[column].dtype)]
    non_numeric_columns = [column for column in columns if column not in numeric_columns]
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

    def project(self, frame: pd.DataFrame, request: ProjectionRequest, version: str) -> dict:
        columns = [column for column in request.feature_columns if column in frame.columns]
        if not columns:
            raise ValueError("No valid feature columns were selected.")
        if request.method == "pca" and len(columns) < 2:
            raise ValueError("PCA requires at least two feature columns.")
        key_data = request.model_dump() | {"version": version}
        key_data["categorical_filters"] = {
            column: sorted(set(values))
            for column, values in request.categorical_filters.items()
        }
        key = hashlib.sha1(json.dumps(key_data, sort_keys=True).encode("utf-8")).hexdigest()
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
            candidate = candidate[candidate[column].fillna("Missing").astype(str).isin(values)]
        if request.method == "lda":
            if not request.color_column or request.color_column not in candidate.columns:
                raise ValueError("LDA requires a valid class/group column in Color by.")
            candidate = candidate[candidate[request.color_column].notna()]
        if request.max_rows_per_class is not None and request.color_column not in candidate.columns:
            raise ValueError("Per-class limit requires a valid Color by column.")
        candidate = candidate.loc[_complete_feature_mask(candidate, columns)]
        available_labels = _labels(candidate, request.color_column)
        available_counts = available_labels.value_counts().to_dict()
        working = _limit_per_class(
            candidate, request.color_column, request.max_rows_per_class, request.random_state
        )
        working = _stable_sample(working, request.max_rows, request.random_state)
        matrix = working[columns].to_numpy(dtype=np.float32)
        if len(matrix) < 3:
            raise ValueError("At least three complete feature rows are required.")
        if request.scale:
            matrix = StandardScaler().fit_transform(matrix)
        subtitle = ""
        if request.method == "pca":
            reducer = PCA(n_components=2, random_state=request.random_state)
            coords = reducer.fit_transform(matrix)
            subtitle = "Explained variance: " + ", ".join(f"{value:.1%}" for value in reducer.explained_variance_ratio_)
        elif request.method == "tsne":
            if len(matrix) > MAX_TSNE_ROWS:
                raise ValueError(f"t-SNE is limited to {MAX_TSNE_ROWS} rows.")
            perplexity = min(request.perplexity, max(2, len(matrix) - 1))
            coords = TSNE(
                n_components=2, perplexity=perplexity, init="pca", learning_rate="auto",
                random_state=request.random_state,
            ).fit_transform(matrix)
        elif request.method == "umap":
            if len(matrix) > MAX_UMAP_ROWS:
                raise ValueError(f"UMAP is limited to {MAX_UMAP_ROWS} rows.")
            try:
                from umap import UMAP
            except ImportError as exc:
                raise ValueError("UMAP requires the umap-learn package.") from exc
            coords = UMAP(
                n_components=2,
                n_neighbors=min(request.umap_neighbors, max(2, len(matrix) - 1)),
                min_dist=request.umap_min_dist,
                random_state=request.random_state,
                n_jobs=1,
            ).fit_transform(matrix)
        else:
            target = working[request.color_column].astype(str)
            class_count = target.nunique()
            if class_count < 2:
                raise ValueError("LDA requires at least two classes after filtering.")
            if len(working) <= class_count:
                raise ValueError("LDA requires more displayed rows than classes; increase the per-class limit.")
            component_count = min(2, class_count - 1, matrix.shape[1])
            coords = LinearDiscriminantAnalysis(n_components=component_count).fit_transform(matrix, target)
            if component_count == 1:
                coords = np.column_stack([coords[:, 0], np.zeros(len(coords), dtype=np.float32)])

        displayed_labels = _labels(working, request.color_column)
        displayed_counts = displayed_labels.value_counts().to_dict()
        points = []
        for position, (row_id, label) in enumerate(
            zip(working["__row_id"].tolist(), displayed_labels.tolist())
        ):
            points.append({
                "x": float(coords[position, 0]),
                "y": float(coords[position, 1]),
                "row_id": int(row_id),
                "label": label,
            })
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
            "subtitle": subtitle,
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
