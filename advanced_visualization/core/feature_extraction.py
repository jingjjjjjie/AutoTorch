"""Forward-pass feature extraction for prepared visualization CSVs."""

from __future__ import annotations

from pathlib import Path
import os
import re

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from advanced_visualization.core.config import ModelRunConfig
from advanced_visualization.models.gradcam import load_gradcam_bundle


FEATURE_PREFIX = "feature_"
FEATURE_COLUMN = re.compile(r"(?:^|_)feature_(\d+)$")


class ImageCsvDataset(Dataset):
    """Image dataset that keeps CSV row positions stable."""

    def __init__(self, df: pd.DataFrame, image_column: str, transform) -> None:
        self.df = df.reset_index(drop=True)
        self.image_column = image_column
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        image_path = self.df.at[index, self.image_column]
        image = Image.open(image_path).convert("RGB")
        return self.transform(image), index


def build_feature_frame(features: np.ndarray) -> pd.DataFrame:
    columns = [f"{FEATURE_PREFIX}{index:04d}" for index in range(features.shape[1])]
    return pd.DataFrame(features, columns=columns)


def extract_features_and_predictions(
    *,
    config: ModelRunConfig,
    csv_path: Path,
    image_column: str,
    output_csv: Path | None = None,
    batch_size: int = 8,
    num_workers: int = 4,
    incremental_from: Path | None = None,
) -> Path:
    df = pd.read_csv(csv_path, low_memory=False)
    if image_column not in df.columns:
        raise ValueError(f"Missing image column {image_column!r} in {csv_path}")

    existing = None
    if incremental_from is not None and incremental_from.is_file():
        existing = pd.read_csv(incremental_from, low_memory=False)

    reusable_features: dict[str, list[str]] = {}
    if existing is not None and image_column in existing.columns:
        for column in existing.columns:
            match = FEATURE_COLUMN.search(str(column))
            if match:
                reusable_features[match.group(1)] = [str(column)]

    feature_columns = [
        reusable_features[index][0]
        for index in sorted(reusable_features, key=lambda value: int(value))
    ]
    keyed_existing = None
    if existing is not None and image_column in existing.columns:
        keyed_existing = existing.drop_duplicates(image_column, keep="last").set_index(
            existing.drop_duplicates(image_column, keep="last")[image_column].astype(str)
        )

    reusable = pd.Series(False, index=df.index)
    if keyed_existing is not None and feature_columns:
        source_keys = df[image_column].astype(str)
        reusable = source_keys.isin(keyed_existing.index)
        candidate = keyed_existing.reindex(source_keys)[feature_columns]
        reusable &= candidate.notna().all(axis=1).to_numpy()

    prediction_column = config.prediction_column
    prediction_values = pd.to_numeric(
        df.get(prediction_column, pd.Series(np.nan, index=df.index)),
        errors="coerce",
    )
    if keyed_existing is not None and prediction_column in keyed_existing.columns:
        old_predictions = pd.to_numeric(
            keyed_existing.reindex(df[image_column].astype(str))[prediction_column],
            errors="coerce",
        ).reset_index(drop=True)
        prediction_values = prediction_values.fillna(old_predictions)

    needs_inference = ~reusable | prediction_values.isna()
    inference_df = df.loc[needs_inference].copy()
    print(
        f"{config.key}: reusing {int((~needs_inference).sum())} rows and "
        f"extracting {len(inference_df)} rows.",
        flush=True,
    )

    bundle = load_gradcam_bundle(config.key)
    model = bundle.model
    transform = bundle.transform
    device = bundle.device
    dataset = ImageCsvDataset(
        inference_df, image_column=image_column, transform=transform
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    features, predictions = _batched_features(model, loader, device)
    if (
        len(inference_df)
        and feature_columns
        and features.shape[1] != len(feature_columns)
    ):
        raise ValueError(
            f"Existing feature width {len(feature_columns)} does not match "
            f"model output width {features.shape[1]}."
        )
    feature_width = features.shape[1] if len(features) else len(feature_columns)
    normalized_columns = [
        f"{FEATURE_PREFIX}{index:04d}" for index in range(feature_width)
    ]
    feature_matrix = np.full((len(df), feature_width), np.nan, dtype=np.float32)
    if keyed_existing is not None and feature_columns:
        old = keyed_existing.reindex(df[image_column].astype(str))[feature_columns]
        feature_matrix[:, :] = old.to_numpy(dtype=np.float32)
    if len(inference_df):
        feature_matrix[needs_inference.to_numpy(), :] = features
        prediction_values.loc[needs_inference] = predictions
    feature_df = pd.DataFrame(feature_matrix, columns=normalized_columns)
    existing_feature_columns = [
        column for column in df.columns if str(column).startswith(FEATURE_PREFIX)
    ]
    if existing_feature_columns:
        df = df.drop(columns=existing_feature_columns)

    output_df = pd.concat([df.reset_index(drop=True), feature_df], axis=1)
    if prediction_column:
        output_df[prediction_column] = prediction_values.to_numpy(dtype=np.float32)

    output_path = output_csv or csv_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    output_df.to_csv(temporary, index=False)
    os.replace(temporary, output_path)
    return output_path


def _batched_features(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    feature_extractor = model.feature_extractor
    features_by_index: dict[int, np.ndarray] = {}
    predictions_by_index: dict[int, float] = {}

    with torch.inference_mode():
        for images, indexes in tqdm(loader, desc="Extracting features", unit="batch"):
            images = images.to(device, non_blocking=True)
            batch_features_tensor = feature_extractor(images)
            batch_outputs = model.mlp_head(batch_features_tensor).squeeze(1)
            batch_probs = _probabilities(batch_outputs)
            batch_features = batch_features_tensor.detach().cpu().numpy()
            for row_index, feature, probability in zip(
                indexes.tolist(), batch_features, batch_probs.detach().cpu().tolist()
            ):
                features_by_index[row_index] = feature.astype(np.float32, copy=False)
                predictions_by_index[row_index] = float(probability)

    if not len(loader.dataset):
        return np.empty((0, 0), dtype=np.float32), np.empty(0, dtype=np.float32)
    features = np.stack([features_by_index[index] for index in range(len(loader.dataset))])
    predictions = np.array(
        [predictions_by_index[index] for index in range(len(loader.dataset))],
        dtype=np.float32,
    )
    return features, predictions


def _probabilities(outputs: torch.Tensor) -> torch.Tensor:
    if outputs.min().detach().item() >= 0.0 and outputs.max().detach().item() <= 1.0:
        return outputs
    return torch.sigmoid(outputs)
