"""Feature projection service with bounded in-memory result caching."""
from __future__ import annotations

import hashlib
import json
import threading
from collections import OrderedDict

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from advanced_visualization.web.models import ProjectionRequest


MAX_TSNE_ROWS = 5000


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
        key = hashlib.sha1(json.dumps(key_data, sort_keys=True).encode("utf-8")).hexdigest()
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]

        candidate = frame.head(request.max_rows)
        numeric = candidate[columns].apply(pd.to_numeric, errors="coerce")
        valid = numeric.notna().all(axis=1)
        working = candidate.loc[valid]
        matrix = numeric.loc[valid].to_numpy(dtype=np.float32)
        if len(matrix) < 3:
            raise ValueError("At least three complete feature rows are required.")
        if request.scale:
            matrix = StandardScaler().fit_transform(matrix)
        subtitle = ""
        if request.method == "pca":
            reducer = PCA(n_components=2, random_state=request.random_state)
            coords = reducer.fit_transform(matrix)
            subtitle = "Explained variance: " + ", ".join(f"{value:.1%}" for value in reducer.explained_variance_ratio_)
        else:
            if len(matrix) > MAX_TSNE_ROWS:
                raise ValueError(f"t-SNE is limited to {MAX_TSNE_ROWS} rows.")
            perplexity = min(request.perplexity, max(2, len(matrix) - 1))
            coords = TSNE(
                n_components=2, perplexity=perplexity, init="pca", learning_rate="auto",
                random_state=request.random_state,
            ).fit_transform(matrix)

        color_column = request.color_column if request.color_column in working.columns else ""
        color_values = working[color_column].tolist() if color_column else ["All"] * len(working)
        points = []
        for position, (row_id, color) in enumerate(zip(working["__row_id"].tolist(), color_values)):
            points.append({
                "x": float(coords[position, 0]),
                "y": float(coords[position, 1]),
                "row_id": int(row_id),
                "label": "Missing" if pd.isna(color) else str(color),
            })
        result = {"method": request.method, "subtitle": subtitle, "rows": len(points), "points": points}
        with self._lock:
            self._cache[key] = result
            self._cache.move_to_end(key)
            while len(self._cache) > self._max_entries:
                self._cache.popitem(last=False)
        return result


projection_service = ProjectionService()
