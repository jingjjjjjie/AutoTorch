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
    image_path_columns,
    infer_standard_columns,
)
from advanced_visualization.core.settings import configured_path, load_settings


FEATURE_PATTERN = __import__("re").compile(
    r"(?:^|_)(feature|feat|embedding|emb)[_-]?\d+$", __import__("re").IGNORECASE
)
EXPLANATION_PATTERN = __import__("re").compile(
    r"(grad.?cam|heatmap|overlay|layer_montage)", __import__("re").IGNORECASE
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
        self._review_cache: OrderedDict[str, tuple[int, int, pd.DataFrame]] = OrderedDict()
        self._schema_cache: dict[str, tuple[int, int, dict]] = {}
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
        if "__row_id" in frame.columns:
            frame = frame.rename(columns={"__row_id": "source___row_id"})
        frame.insert(0, "__row_id", np.arange(len(frame), dtype=np.int64))
        with self._lock:
            self._cache[source_id] = (*signature, frame)
            self._cache.move_to_end(source_id)
            while len(self._cache) > self._max_cached_sources:
                self._cache.popitem(last=False)
        return frame

    def review_dataframe(self, source_id: str) -> pd.DataFrame:
        """Load only columns used by review workflows, excluding feature vectors."""
        source = self.source(source_id)
        stat = source.path.stat()
        signature = (stat.st_mtime_ns, stat.st_size)
        with self._lock:
            cached = self._review_cache.get(source_id)
            if cached and cached[:2] == signature:
                self._review_cache.move_to_end(source_id)
                return cached[2]

        header = pd.read_csv(source.path, nrows=0).columns.astype(str).tolist()
        review_columns = [column for column in header if not FEATURE_PATTERN.search(column)]
        frame = pd.read_csv(source.path, usecols=review_columns, low_memory=False)
        frame.columns = frame.columns.astype(str)
        if "__row_id" in frame.columns:
            frame = frame.rename(columns={"__row_id": "source___row_id"})
        frame.insert(0, "__row_id", np.arange(len(frame), dtype=np.int64))
        with self._lock:
            self._review_cache[source_id] = (*signature, frame)
            self._schema_cache.pop(source_id, None)
            self._review_cache.move_to_end(source_id)
            while len(self._review_cache) > self._max_cached_sources:
                self._review_cache.popitem(last=False)
        return frame

    def schema(self, source_id: str) -> dict:
        source = self.source(source_id)
        frame = self.review_dataframe(source_id)
        stat = source.path.stat()
        signature = (stat.st_mtime_ns, stat.st_size)
        with self._lock:
            cached = self._schema_cache.get(source_id)
            if cached and cached[:2] == signature:
                return cached[2]

        public = frame.drop(columns="__row_id")
        columns = public.columns.tolist()
        settings = load_settings()
        model = next((item for item in settings.models if item.key == source.model_key), None)
        defaults = infer_standard_columns(public, model.prediction_column if model else "")
        categorical = self._categorical_columns(public, features=set())
        images = image_path_columns(public)
        availability = {column: self._path_availability(public[column]) for column in images}
        images.sort(key=lambda column: (availability[column] <= 0, -availability[column], column))
        gradcams = [column for column in images if EXPLANATION_PATTERN.search(column)]
        configured_image = model.image_column if model and model.image_column in images else ""
        inferred_image = defaults.get("image_column", "")
        if configured_image and availability.get(configured_image, 0) > 0:
            defaults["image_column"] = configured_image
        elif availability.get(inferred_image, 0) <= 0:
            defaults["image_column"] = next(
                (column for column in images if availability[column] > 0 and column not in gradcams),
                next((column for column in images if availability[column] > 0), ""),
            )
        header = pd.read_csv(source.path, nrows=0).columns.astype(str).tolist()
        features = [column for column in header if FEATURE_PATTERN.search(column)]
        categories = {}
        for column in categorical:
            values = public[column].fillna("Missing").astype(str).unique()
            if len(values) <= 200:
                categories[column] = sorted(values.tolist())
        details = {
            "source": source,
            "columns": columns,
            "numeric_columns": public.select_dtypes(include=[np.number]).columns.tolist(),
            "categorical_columns": categorical,
            "image_columns": images,
            "gradcam_columns": gradcams,
            "feature_columns": features,
            "defaults": defaults,
            "categories": categories,
            "image_availability": availability,
        }
        with self._lock:
            self._schema_cache[source_id] = (*signature, details)
        return details

    @staticmethod
    def _path_availability(series: pd.Series, sample_size: int = 200) -> float:
        values = series.dropna().astype(str)
        values = values[values.str.strip().ne("")].head(sample_size)
        if values.empty:
            return 0.0
        existing = sum(Path(value).expanduser().is_file() for value in values)
        return existing / len(values)

    @staticmethod
    def _categorical_columns(frame: pd.DataFrame, features: set[str]) -> list[str]:
        """Infer filter columns without scanning high-dimensional feature vectors."""
        limit = min(120, max(12, len(frame) // 8))
        result = []
        for column in frame.select_dtypes(exclude=[np.number]).columns:
            if column in features:
                continue
            if frame[column].nunique(dropna=True) <= limit:
                result.append(column)
        for column in frame.select_dtypes(include=[np.number]).columns:
            if column in features:
                continue
            if frame[column].nunique(dropna=True) <= limit:
                result.append(column)
        return result


repository = DatasetRepository()
