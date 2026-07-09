"""Export AutoTorch embeddings to a CSV for interactive visualization."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/autotorch_feature_visualization_mpl")

USER_SITE = Path.home() / ".local" / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
user_site_was_enabled = str(USER_SITE) in sys.path
if user_site_was_enabled:
    sys.path.remove(str(USER_SITE))

import numpy as np
import pandas as pd

if USER_SITE.exists() and str(USER_SITE) not in sys.path:
    sys.path.append(str(USER_SITE))

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data.idfraud.transforms import build_transform
from models import build_model


DEFAULT_MEAN = (0.485, 0.456, 0.406)
DEFAULT_STD = (0.229, 0.224, 0.225)
FEATURE_PREFIX = "feature_"


class ImageCsvDataset(Dataset):
    """Image dataset that keeps source CSV row indexes stable."""

    def __init__(self, df: pd.DataFrame, image_column: str, transform) -> None:
        self.df = df.reset_index(drop=True)
        self.image_column = image_column
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image_path = self.df.at[index, self.image_column]
        image = Image.open(image_path).convert("RGB")
        return self.transform(image), index


def parse_float_triplet(raw: str) -> tuple[float, float, float]:
    values = tuple(float(part.strip()) for part in raw.split(","))
    if len(values) != 3:
        raise argparse.ArgumentTypeError("Expected exactly three comma-separated floats.")
    return values


def load_state_dict(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    normalized = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        normalized[key] = value

    model.load_state_dict(normalized, strict=True)


def batched_features(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    output_type: str,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    feature_extractor = model.feature_extractor
    features_by_index: dict[int, np.ndarray] = {}
    predictions_by_index: dict[int, float] = {}

    with torch.inference_mode():
        for images, indexes in tqdm(loader, desc="Extracting features"):
            images = images.to(device, non_blocking=True)
            batch_features_tensor = feature_extractor(images)
            batch_outputs = model.mlp_head(batch_features_tensor).squeeze(1)
            batch_probs = batch_outputs if output_type == "probs" else torch.sigmoid(batch_outputs)
            batch_features = batch_features_tensor.detach().cpu().numpy()
            batch_probs = batch_probs.detach().cpu().tolist()
            for row_index, feature, prob in zip(indexes.tolist(), batch_features, batch_probs):
                features_by_index[row_index] = feature.astype(np.float32, copy=False)
                predictions_by_index[row_index] = float(prob)

    features = np.stack([features_by_index[index] for index in range(len(loader.dataset))])
    predictions = np.array([predictions_by_index[index] for index in range(len(loader.dataset))], dtype=np.float32)
    return features, predictions


def build_feature_frame(features: np.ndarray) -> pd.DataFrame:
    columns = [f"{FEATURE_PREFIX}{index:04d}" for index in range(features.shape[1])]
    return pd.DataFrame(features, columns=columns)


def validate_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}. Available columns: {list(df.columns)}")


def limit_class_rows(
    df: pd.DataFrame,
    class_column: str | None,
    class_value: str,
    max_rows: int | None,
    random_state: int,
) -> pd.DataFrame:
    if not class_column or max_rows is None:
        return df

    validate_columns(df, [class_column])
    class_mask = df[class_column].astype(str).str.lower().eq(class_value.lower())
    class_df = df[class_mask]
    if len(class_df) <= max_rows:
        return df.reset_index(drop=True)

    sampled = class_df.sample(n=max_rows, random_state=random_state)
    limited = pd.concat([sampled, df[~class_mask]], axis=0).sort_index().reset_index(drop=True)
    print(f"Sampled {class_value} rows from {len(class_df):,} to {max_rows:,}; extracting {len(limited):,} total rows.")
    return limited


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, type=Path, help="Input CSV containing image paths and metadata.")
    parser.add_argument("--checkpoint", required=True, type=Path, help="Trained AutoTorch model checkpoint.")
    parser.add_argument("--output", required=True, type=Path, help="Output feature CSV path.")
    parser.add_argument("--model-name", default="unireplknet_t", help="Backbone name passed to AutoTorch build_model.")
    parser.add_argument("--image-column", default="path", help="Column containing image paths.")
    parser.add_argument("--label-column", default="label", help="Metadata label column to validate and preserve.")
    parser.add_argument("--head-type", default="legacy_v2", help="Classifier head used by the checkpoint.")
    parser.add_argument("--output-type", default="probs", choices=("probs", "logits"), help="Model output interpretation.")
    parser.add_argument("--prediction-column", default="unireplknet_t_pred", help="Output prediction probability column.")
    parser.add_argument("--image-size", default=1024, type=int, help="Square input image size.")
    parser.add_argument("--transform-version", default="v1", choices=("v1", "v2", "v3"), help="AutoTorch transform version.")
    parser.add_argument("--batch-size", default=8, type=int, help="Inference batch size.")
    parser.add_argument("--num-workers", default=4, type=int, help="DataLoader workers.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Torch device.")
    parser.add_argument("--normalize-mean", default=DEFAULT_MEAN, type=parse_float_triplet)
    parser.add_argument("--normalize-std", default=DEFAULT_STD, type=parse_float_triplet)
    parser.add_argument("--limit-class-column", default=None, help="Class column to sample before inference.")
    parser.add_argument("--limit-class-value", default="Genuine", help="Class value to sample before inference.")
    parser.add_argument("--limit-class-rows", default=None, type=int, help="Maximum sampled rows for --limit-class-value.")
    parser.add_argument("--random-state", default=42, type=int, help="Random seed for class sampling.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.csv)
    validate_columns(df, [args.image_column, args.label_column])
    df = limit_class_rows(
        df,
        class_column=args.limit_class_column,
        class_value=args.limit_class_value,
        max_rows=args.limit_class_rows,
        random_state=args.random_state,
    )

    device = torch.device(args.device)
    model = build_model(
        model_name=args.model_name,
        device=device,
        task="classification",
        head_type=args.head_type,
        freeze_backbone=False,
    )
    load_state_dict(model, args.checkpoint, device)

    transform = build_transform(
        image_size=args.image_size,
        normalize_mean=args.normalize_mean,
        normalize_std=args.normalize_std,
        version=args.transform_version,
    )
    dataset = ImageCsvDataset(df, image_column=args.image_column, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    features, predictions = batched_features(model, loader, device, output_type=args.output_type)
    output_df = pd.concat([df.reset_index(drop=True), build_feature_frame(features)], axis=1)
    output_df[args.prediction_column] = predictions

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(args.output, index=False)
    print(f"Wrote {len(output_df)} rows, {features.shape[1]} features, and predictions to {args.output}")


if __name__ == "__main__":
    main()
