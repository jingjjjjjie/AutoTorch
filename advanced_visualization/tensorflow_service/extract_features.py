"""Batch-export registered TensorFlow VAN Small predictions and avg-pool features."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image, ImageOps

from advanced_visualization.core.model_router import registered_model_routes
from advanced_visualization.tensorflow_service.app import (
    _custom_objects,
    _prepare_image,
)


FEATURE_PREFIX = "feature_"
FEATURE_COLUMN = re.compile(r"(?:^|_)feature_(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--image-column", required=True)
    parser.add_argument("--prediction-column", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--incremental-from", type=Path)
    return parser.parse_args()


def load_feature_model(model_id: str) -> tuple[object, tf.keras.Model]:
    routes = registered_model_routes()
    if model_id not in routes:
        raise ValueError(f"Unknown registered model: {model_id}")
    route = routes[model_id]
    if route.framework != "tensorflow" or route.checkpoint is None:
        raise ValueError(f"{model_id} is not a checkpoint-backed TensorFlow model.")
    model = tf.keras.models.load_model(
        str(route.checkpoint), custom_objects=_custom_objects(), compile=False
    )
    feature_layer = model.get_layer("avg_pool")
    return route, tf.keras.Model(
        model.inputs,
        [feature_layer.output, model.output],
        name=f"{model_id}_feature_export",
    )


def load_image(path: str, image_size: int) -> tf.Tensor:
    with Image.open(path) as opened:
        transposed = ImageOps.exif_transpose(opened)
        image = (transposed or opened).convert("RGB")
        return _prepare_image(image, image_size)[0]


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be at least 1.")
    frame = pd.read_csv(args.csv, low_memory=False)
    if (
        args.image_column == "absolute_crop_path"
        and args.image_column not in frame.columns
        and "absolute_ocr_path" in frame.columns
    ):
        frame[args.image_column] = frame["absolute_ocr_path"]
    if args.image_column not in frame.columns:
        raise ValueError(f"Missing image column: {args.image_column}")
    existing = None
    if args.incremental_from is not None and args.incremental_from.is_file():
        existing = pd.read_csv(args.incremental_from, low_memory=False)
    feature_columns = []
    if existing is not None:
        numbered = []
        for column in existing.columns:
            match = FEATURE_COLUMN.search(str(column))
            if match:
                numbered.append((int(match.group(1)), str(column)))
        feature_columns = [column for _, column in sorted(numbered)]
    keyed_existing = None
    if existing is not None and args.image_column in existing.columns:
        unique = existing.drop_duplicates(args.image_column, keep="last")
        keyed_existing = unique.set_index(unique[args.image_column].astype(str))
    source_keys = frame[args.image_column].astype(str)
    reusable = pd.Series(False, index=frame.index)
    if keyed_existing is not None and feature_columns:
        old = keyed_existing.reindex(source_keys)[feature_columns]
        reusable = old.notna().all(axis=1).reset_index(drop=True)
    prediction_values = pd.to_numeric(
        frame.get(args.prediction_column, pd.Series(np.nan, index=frame.index)),
        errors="coerce",
    )
    if keyed_existing is not None and args.prediction_column in keyed_existing.columns:
        old_predictions = pd.to_numeric(
            keyed_existing.reindex(source_keys)[args.prediction_column], errors="coerce"
        ).reset_index(drop=True)
        prediction_values = prediction_values.fillna(old_predictions)
    needs_inference = ~reusable | prediction_values.isna()
    inference_frame = frame.loc[needs_inference].copy()
    print(
        f"{args.model_id}: reusing {int((~needs_inference).sum())} rows and "
        f"extracting {len(inference_frame)} rows.",
        flush=True,
    )
    route, model = load_feature_model(args.model_id)

    feature_batches: list[np.ndarray] = []
    prediction_batches: list[np.ndarray] = []
    total = len(inference_frame)
    for start in range(0, total, args.batch_size):
        stop = min(total, start + args.batch_size)
        images = tf.stack(
            [
                load_image(str(path), route.image_size)
                for path in inference_frame[args.image_column].iloc[start:stop]
            ]
        )
        features, predictions = model(images, training=False)
        feature_batches.append(np.asarray(features, dtype=np.float32))
        prediction_batches.append(
            np.asarray(predictions, dtype=np.float32).reshape(-1)
        )
        print(f"{args.model_id}: {stop}/{total}", flush=True)

    if feature_batches:
        features = np.concatenate(feature_batches, axis=0)
        predictions = np.concatenate(prediction_batches, axis=0)
    else:
        features = np.empty((0, len(feature_columns)), dtype=np.float32)
        predictions = np.empty(0, dtype=np.float32)
    if feature_columns and features.shape[1] != len(feature_columns):
        raise ValueError(
            f"Existing feature width {len(feature_columns)} does not match "
            f"model output width {features.shape[1]}."
        )
    feature_matrix = np.full((len(frame), features.shape[1]), np.nan, dtype=np.float32)
    if keyed_existing is not None and feature_columns:
        feature_matrix[:, :] = keyed_existing.reindex(source_keys)[feature_columns].to_numpy(
            dtype=np.float32
        )
    feature_matrix[needs_inference.to_numpy(), :] = features
    prediction_values.loc[needs_inference] = predictions
    old_features = [
        column for column in frame.columns if str(column).startswith(FEATURE_PREFIX)
    ]
    if old_features:
        frame = frame.drop(columns=old_features)
    feature_frame = pd.DataFrame(
        feature_matrix,
        columns=[f"{FEATURE_PREFIX}{index:04d}" for index in range(features.shape[1])],
    )
    output = pd.concat([frame.reset_index(drop=True), feature_frame], axis=1)
    output[args.prediction_column] = prediction_values.to_numpy(dtype=np.float32)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    output.to_csv(temporary, index=False)
    os.replace(temporary, args.output)
    print(
        f"Wrote {len(output)} rows, {features.shape[1]} features, and "
        f"{args.prediction_column} to {args.output}"
    )


if __name__ == "__main__":
    main()
