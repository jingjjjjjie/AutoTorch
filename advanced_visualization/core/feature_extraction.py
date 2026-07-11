"""Forward-pass feature extraction for prepared visualization CSVs."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from advanced_visualization.core.config import ModelRunConfig
from advanced_visualization.models.gradcam import load_gradcam_bundle


FEATURE_PREFIX = "feature_"


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
) -> Path:
    df = pd.read_csv(csv_path, low_memory=False)
    if image_column not in df.columns:
        raise ValueError(f"Missing image column {image_column!r} in {csv_path}")

    bundle = load_gradcam_bundle(config.key)
    model = bundle["model"]
    transform = bundle["transform"]
    device = bundle["device"]
    dataset = ImageCsvDataset(df, image_column=image_column, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    features, predictions = _batched_features(model, loader, device)
    feature_df = build_feature_frame(features)
    existing_feature_columns = [column for column in df.columns if str(column).startswith(FEATURE_PREFIX)]
    if existing_feature_columns:
        df = df.drop(columns=existing_feature_columns)

    output_df = pd.concat([df.reset_index(drop=True), feature_df], axis=1)
    if config.prediction_column:
        output_df[config.prediction_column] = predictions

    output_path = output_csv or csv_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)
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
            for row_index, feature, probability in zip(indexes.tolist(), batch_features, batch_probs.detach().cpu().tolist()):
                features_by_index[row_index] = feature.astype(np.float32, copy=False)
                predictions_by_index[row_index] = float(probability)

    features = np.stack([features_by_index[index] for index in range(len(loader.dataset))])
    predictions = np.array([predictions_by_index[index] for index in range(len(loader.dataset))], dtype=np.float32)
    return features, predictions


def _probabilities(outputs: torch.Tensor) -> torch.Tensor:
    if outputs.min().detach().item() >= 0.0 and outputs.max().detach().item() <= 1.0:
        return outputs
    return torch.sigmoid(outputs)
