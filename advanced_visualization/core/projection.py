"""Framework-independent feature projection algorithms."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/autotorch_visualization_numba")

import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


ProjectionMethod = Literal["pca", "tsne", "umap", "lda"]
MAX_TSNE_ROWS = 5_000
MAX_UMAP_ROWS = 50_000


@dataclass(frozen=True)
class ProjectionParameters:
    method: ProjectionMethod = "pca"
    scale: bool = True
    perplexity: int = 30
    umap_neighbors: int = 15
    umap_min_dist: float = 0.1
    random_state: int = 42


@dataclass(frozen=True)
class ProjectionCoordinates:
    values: np.ndarray
    note: str


def project_matrix(
    matrix: np.ndarray,
    parameters: ProjectionParameters,
    *,
    labels: np.ndarray | None = None,
) -> ProjectionCoordinates:
    """Project a complete numeric matrix to two dimensions."""
    values = np.asarray(matrix, dtype=np.float32)
    if values.ndim != 2 or len(values) < 3:
        raise ValueError("At least three complete feature rows are required.")
    if values.shape[1] < 1:
        raise ValueError("At least one feature column is required.")
    if parameters.method == "pca" and values.shape[1] < 2:
        raise ValueError("PCA requires at least two feature columns.")
    if parameters.scale:
        values = StandardScaler().fit_transform(values)

    if parameters.method == "pca":
        reducer = PCA(n_components=2, random_state=parameters.random_state)
        coordinates = reducer.fit_transform(values)
        note = "Explained variance: " + ", ".join(
            f"{value:.1%}" for value in reducer.explained_variance_ratio_
        )
    elif parameters.method == "tsne":
        if len(values) > MAX_TSNE_ROWS:
            raise ValueError(f"t-SNE is limited to {MAX_TSNE_ROWS:,} rows.")
        perplexity = min(parameters.perplexity, max(2, len(values) - 1))
        coordinates = TSNE(
            n_components=2,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            random_state=parameters.random_state,
        ).fit_transform(values)
        note = f"Perplexity: {perplexity}"
    elif parameters.method == "umap":
        if len(values) > MAX_UMAP_ROWS:
            raise ValueError(f"UMAP is limited to {MAX_UMAP_ROWS:,} rows.")
        try:
            from umap import UMAP
        except ImportError as exc:
            raise ValueError("UMAP requires the umap-learn package.") from exc
        neighbors = min(max(2, parameters.umap_neighbors), max(2, len(values) - 1))
        coordinates = UMAP(
            n_components=2,
            n_neighbors=neighbors,
            min_dist=parameters.umap_min_dist,
            metric="euclidean",
            random_state=parameters.random_state,
            n_jobs=1,
        ).fit_transform(values)
        note = f"Neighbors: {neighbors}; min_dist: {parameters.umap_min_dist:.2f}"
    else:
        if labels is None:
            raise ValueError("LDA requires class labels.")
        target = np.asarray(labels).astype(str)
        class_count = len(np.unique(target))
        if class_count < 2:
            raise ValueError("LDA requires at least two classes after filtering.")
        if len(values) <= class_count:
            raise ValueError("LDA requires more displayed rows than classes.")
        component_count = min(2, class_count - 1, values.shape[1])
        coordinates = LinearDiscriminantAnalysis(
            n_components=component_count
        ).fit_transform(values, target)
        if component_count == 1:
            coordinates = np.column_stack(
                [coordinates[:, 0], np.zeros(len(coordinates), dtype=np.float32)]
            )
        note = f"Target classes: {class_count}; axes: {component_count}"

    return ProjectionCoordinates(np.asarray(coordinates, dtype=np.float32), note)
