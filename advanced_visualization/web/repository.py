"""Source discovery and modification-aware dataframe caching."""
from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from advanced_visualization.core.artifacts import available_data_sources
from advanced_visualization.core.columns import (
    GRADCAM_PATTERN,
    categorical_columns,
    image_path_columns,
    infer_standard_columns,
)
from advanced_visualization.core.settings import configured_path, load_settings


FEATURE_PATTERN = __import__("re").compile(
    r"(?:^|_)(feature|feat|embedding|emb)[_-]?\d+$", __import__("re").IGNORECASE
)


@dataclass(frozen=True)
class DataSource:
    id: str
    label: str
    path: Path
    model_key: str
    artifact_dir: Path | None


def _source_id(path: Path, model_key: str) -> str:
    raw = f"{path.expanduser().resolve()}:{model_key}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


class DatasetRepository:
    """Owns loaded dataframes and invalidates them when their CSV changes."""

    def __init__(self, max_cached_sources: int = 4) -> None:
        self._max_cached_sources = max_cached_sources
        self._cache: OrderedDict[str, tuple[int, int, pd.DataFrame]] = OrderedDict()
        self._lock = threading.RLock()

    def sources(self) -> list[DataSource]:
        result = []
        seen: set[Path] = set()
        for raw in available_data_sources():
            path = Path(raw["path"]).expanduser()
            model_key = str(raw.get("model_key") or "")
            artifact = raw.get("artifact_dir")
            result.append(
                DataSource(
                    id=_source_id(path, model_key),
                    label=str(raw["label"]),
                    path=path,
                    model_key=model_key,
                    artifact_dir=Path(artifact).expanduser() if artifact else None,
                )
            )
            seen.add(path.resolve())
        for model in load_settings().models:
            if not model.enabled or not model.feature_csv.strip():
                continue
            path = configured_path(model.feature_csv)
            if path.resolve() in seen:
                continue
            result.append(
                DataSource(
                    id=_source_id(path, model.key),
                    label=f"{model.key} - features",
                    path=path,
                    model_key=model.key,
                    artifact_dir=configured_path(model.artifact_dir) if model.artifact_dir else None,
                )
            )
            seen.add(path.resolve())
        return result

    def source(self, source_id: str) -> DataSource:
        source = next((item for item in self.sources() if item.id == source_id), None)
        if source is None:
            raise KeyError(f"Unknown data source: {source_id}")
        if not source.path.is_file():
            raise FileNotFoundError(f"CSV does not exist: {source.path}")
        return source

    def dataframe(self, source_id: str) -> pd.DataFrame:
        source = self.source(source_id)
        stat = source.path.stat()
        signature = (stat.st_mtime_ns, stat.st_size)
        with self._lock:
            cached = self._cache.get(source_id)
            if cached and cached[:2] == signature:
                self._cache.move_to_end(source_id)
                return cached[2]

        header = pd.read_csv(source.path, nrows=0).columns.astype(str)
        feature_dtypes = {column: np.float32 for column in header if FEATURE_PATTERN.search(column)}
        frame = pd.read_csv(source.path, low_memory=False, dtype=feature_dtypes)
        frame.columns = frame.columns.astype(str)
        frame.insert(0, "__row_id", np.arange(len(frame), dtype=np.int64))
        with self._lock:
            self._cache[source_id] = (*signature, frame)
            self._cache.move_to_end(source_id)
            while len(self._cache) > self._max_cached_sources:
                self._cache.popitem(last=False)
        return frame

    def schema(self, source_id: str) -> dict:
        source = self.source(source_id)
        frame = self.dataframe(source_id)
        public = frame.drop(columns="__row_id")
        columns = public.columns.tolist()
        settings = load_settings()
        model = next((item for item in settings.models if item.key == source.model_key), None)
        defaults = infer_standard_columns(public, model.prediction_column if model else "")
        categorical = categorical_columns(public)
        images = image_path_columns(public)
        gradcams = [column for column in images if GRADCAM_PATTERN.search(column)]
        features = [
            column for column in public.select_dtypes(include=[np.number]).columns
            if FEATURE_PATTERN.search(column)
        ]
        categories = {}
        for column in categorical:
            values = public[column].fillna("Missing").astype(str).unique()
            if len(values) <= 200:
                categories[column] = sorted(values.tolist())
        return {
            "source": source,
            "columns": columns,
            "numeric_columns": public.select_dtypes(include=[np.number]).columns.tolist(),
            "categorical_columns": categorical,
            "image_columns": images,
            "gradcam_columns": gradcams,
            "feature_columns": features,
            "defaults": defaults,
            "categories": categories,
        }


repository = DatasetRepository()
