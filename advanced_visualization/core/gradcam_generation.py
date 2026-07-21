"""Batch Grad-CAM generation independent of command-line parsing."""

from __future__ import annotations

import os
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

os.environ.setdefault("MPLCONFIGDIR", "/tmp/autotorch_gradcam_mpl")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from advanced_visualization.core.artifacts import (
    default_csv_paths as artifact_csv_paths,
)
from advanced_visualization.core.artifacts import load_manifest
from advanced_visualization.core.config import (
    DEFAULT_GRADCAM_ROOT,
    IMAGE_COLUMNS,
    all_model_runs,
    gradcam_artifact_root,
)
from advanced_visualization.core.gradcam_cache import (
    gradcam_cache_candidates,
    gradcam_roots,
)
from advanced_visualization.core.heatmap import jet_overlay
from advanced_visualization.core.images import valid_image
from advanced_visualization.core.settings import configured_path, load_settings

if TYPE_CHECKING:
    from advanced_visualization.models.base import GradcamBundle


DEFAULT_IMAGE_COLUMNS = IMAGE_COLUMNS


@dataclass(frozen=True)
class GradcamGenerationOptions:
    """Validated options for one Grad-CAM generation run."""

    image_column: str | None = None
    filters: tuple[str, ...] = ()
    offset: int = 0
    limit: int | None = None
    num_shards: int = 1
    shard_index: int = 0
    batch_size: int = 32
    num_workers: int = min(8, os.cpu_count() or 1)
    prefetch_factor: int = 4
    save_workers: int = 8
    max_output_side: int | None = None
    output_root: Path | None = None
    cam_methods: tuple[str, ...] = ("gradcam",)
    cam_targets: tuple[str, ...] = ("fraud",)
    max_error_examples: int = 10
    only_missing: bool = True
    dry_run: bool = False
    stop_on_error: bool = False

    def validate(self) -> None:
        if self.offset < 0:
            raise ValueError("offset cannot be negative.")
        if self.limit is not None and self.limit < 0:
            raise ValueError("limit cannot be negative.")
        if self.num_shards < 1:
            raise ValueError("num_shards must be at least 1.")
        if not 0 <= self.shard_index < self.num_shards:
            raise ValueError("shard_index must be in [0, num_shards).")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1.")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative.")
        if self.prefetch_factor < 1:
            raise ValueError("prefetch_factor must be at least 1.")
        if self.save_workers < 1:
            raise ValueError("save_workers must be at least 1.")
        if self.max_error_examples < 0:
            raise ValueError("max_error_examples cannot be negative.")
        if self.max_output_side is not None and self.max_output_side < 1:
            raise ValueError("max_output_side must be at least 1 when provided.")
        if not self.cam_methods or not set(self.cam_methods) <= {
            "gradcam",
            "gradcam++",
        }:
            raise ValueError("cam_methods must contain gradcam and/or gradcam++.")
        if not self.cam_targets or not set(self.cam_targets) <= {"fraud", "genuine"}:
            raise ValueError("cam_targets must contain fraud and/or genuine.")


def config_key_for_csv(csv_path: Path) -> str:
    manifest = load_manifest(csv_path.parent)
    if manifest and manifest.prepared_csv.resolve() == csv_path.resolve():
        return manifest.model_key
    for model in load_settings().models:
        if not model.enabled or not model.artifact_dir.strip():
            continue
        artifact_dir = configured_path(model.artifact_dir)
        if (
            csv_path.resolve() == (artifact_dir / "prepared_predictions.csv").resolve()
            or csv_path.parent.resolve() == artifact_dir.resolve()
        ):
            return model.key
    return csv_path.stem


def default_csv_paths() -> list[Path]:
    model_runs = all_model_runs()
    return [
        path for path in artifact_csv_paths() if config_key_for_csv(path) in model_runs
    ]


def resolve_csv_paths(raw_paths: list[Path]) -> list[Path]:
    if not raw_paths:
        return default_csv_paths()

    paths: list[Path] = []
    for raw_path in raw_paths:
        path = raw_path.expanduser()
        if path.is_dir():
            manifest = load_manifest(path)
            if manifest and manifest.prepared_csv.exists():
                paths.append(manifest.prepared_csv)
            else:
                model_runs = all_model_runs()
                paths.extend(
                    sorted(
                        candidate
                        for candidate in path.glob("*.csv")
                        if config_key_for_csv(candidate) in model_runs
                    )
                )
        else:
            paths.append(path)
    return paths


def infer_image_column(df: pd.DataFrame, requested: str | None) -> str:
    if requested:
        if requested not in df.columns:
            raise ValueError(
                f"Missing image column {requested!r}. Available columns: {list(df.columns)}"
            )
        return requested
    for column in DEFAULT_IMAGE_COLUMNS:
        if column in df.columns:
            return column
    raise ValueError(f"No image path column found. Tried: {DEFAULT_IMAGE_COLUMNS}")


def apply_filters(df: pd.DataFrame, options: GradcamGenerationOptions) -> pd.DataFrame:
    filtered = df
    for raw_filter in options.filters:
        if "=" not in raw_filter:
            raise ValueError(f"Filter must be COLUMN=VALUE, got: {raw_filter}")
        column, value = raw_filter.split("=", 1)
        column = column.strip()
        value = value.strip()
        if column not in filtered.columns:
            raise ValueError(f"Filter column {column!r} not in CSV.")
        filtered = filtered[filtered[column].fillna("Missing").astype(str).eq(value)]

    if options.offset:
        filtered = filtered.iloc[options.offset :]
    if options.limit is not None:
        filtered = filtered.iloc[: options.limit]
    if options.num_shards > 1:
        filtered = filtered.iloc[options.shard_index :: options.num_shards]
    return filtered


def existing_gradcam_for_image(
    config_key: str, image_path: Path, method: str = "", target: str = "fraud"
) -> Path | None:
    roots = gradcam_roots(config_key, "")
    for root in roots:
        for candidate in gradcam_cache_candidates(
            root, image_path, method=method, target=target
        ):
            if candidate.is_file():
                return candidate
    return None


def output_root_for_config(config_key: str, options: GradcamGenerationOptions) -> Path:
    if options.output_root:
        return options.output_root.expanduser() / config_key
    output_root = gradcam_artifact_root(config_key)
    if output_root is None:
        return DEFAULT_GRADCAM_ROOT / config_key
    return output_root


def output_path_for_image(
    config_key: str,
    image_path: Path,
    options: GradcamGenerationOptions,
    method: str = "gradcam",
    target: str = "fraud",
) -> Path:
    output_root = output_root_for_config(config_key, options)
    output_root.mkdir(parents=True, exist_ok=True)
    return gradcam_cache_candidates(
        output_root, image_path, method=method, target=target
    )[0]


class GradcamDataset(Dataset):
    def __init__(
        self,
        jobs: list[tuple[int, Path, dict[tuple[str, str], Path]]],
        transform,
        max_output_side: int | None = None,
    ):
        self.jobs = jobs
        self.transform = transform
        self.max_output_side = max_output_side

    def __len__(self) -> int:
        return len(self.jobs)

    def __getitem__(self, index: int):
        row_index, image_path, output_paths = self.jobs[index]
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception:
            return None
        tensor = self.transform(image)
        if self.max_output_side and max(image.size) > self.max_output_side:
            image.thumbnail(
                (self.max_output_side, self.max_output_side), Image.Resampling.LANCZOS
            )
        base = np.asarray(image, dtype=np.uint8)
        return (
            tensor,
            base,
            {key: str(path) for key, path in output_paths.items()},
            row_index,
        )


def collate_valid(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    tensors = torch.stack([item[0] for item in batch], dim=0)
    bases = [item[1] for item in batch]
    outputs = [item[2] for item in batch]
    row_indices = [item[3] for item in batch]
    return tensors, bases, outputs, row_indices


def overlay_from_cam(base: np.ndarray, cam: np.ndarray) -> Image.Image:
    return jet_overlay(base, cam)


def process_batch(
    config_key: str,
    bundle: GradcamBundle,
    batch,
    activations,
    gradients,
    save_pool: ThreadPoolExecutor,
    cam_methods: list[str],
    cam_targets: list[str],
):
    from advanced_visualization.models.gradcam import compute_cam, gradcam_score

    tensors, bases, outputs, row_indices = batch
    model = bundle.model
    device = bundle.device

    activations.clear()
    gradients.clear()
    input_tensor = tensors.to(device, non_blocking=True)
    model.zero_grad(set_to_none=True)
    score = gradcam_score(model, input_tensor, config_key=config_key).sum()

    if "value" not in activations:
        raise RuntimeError(
            f"Grad-CAM target layer did not produce activations for rows {row_indices[:5]}"
        )
    activation = activations["value"].detach()
    futures = []
    for target_index, target in enumerate(cam_targets):
        gradients.clear()
        model.zero_grad(set_to_none=True)
        target_score = -score if target == "genuine" else score
        target_score.backward(retain_graph=target_index < len(cam_targets) - 1)
        if "value" not in gradients:
            raise RuntimeError(
                f"Grad-CAM target layer did not produce gradients for rows {row_indices[:5]}"
            )
        gradient = gradients["value"].detach()
        for method in cam_methods:
            cam_batch = compute_cam(
                activation, gradient, config_key=config_key, method=method
            ).float()
            for cam, base, output_paths in zip(cam_batch, bases, outputs):
                output_path = output_paths.get((method, target))
                if not output_path:
                    continue
                h, w = base.shape[:2]
                resized = F.interpolate(
                    cam.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False
                ).squeeze()
                resized = (resized - resized.min()) / (
                    resized.max() - resized.min() + 1e-8
                )
                # Saving runs on background threads, so the array must own a fully
                # completed CPU copy. A non-blocking D2H copy can otherwise race
                cam_np = resized.detach().float().cpu().numpy().copy()
                futures.append(
                    save_pool.submit(_save_overlay, base, cam_np, output_path)
                )
    return futures


def _save_overlay(
    base: np.ndarray, cam: np.ndarray, output_path: str
) -> tuple[bool, str | None]:
    output = Path(output_path)
    tmp_output = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        overlay_from_cam(base, cam).save(tmp_output, format="PNG", compress_level=3)
        os.replace(tmp_output, output)
        return True, None
    except Exception as exc:
        tmp_output.unlink(missing_ok=True)
        return False, f"{type(exc).__name__}: {exc}"


def collect_finished_saves(
    pending: list, block: bool = False
) -> tuple[list, int, int, list[str]]:
    if not pending:
        return pending, 0, 0, []

    if block:
        done, not_done = wait(pending, return_when=FIRST_COMPLETED)
        remaining = list(not_done)
    else:
        done = {fut for fut in pending if fut.done()}
        remaining = [fut for fut in pending if fut not in done]

    generated = 0
    failed = 0
    errors = []
    for fut in done:
        ok, error = fut.result()
        if ok:
            generated += 1
        else:
            failed += 1
            if error:
                errors.append(error)
    return remaining, generated, failed, errors


def pregenerate_csv(
    csv_path: Path, options: GradcamGenerationOptions
) -> tuple[int, int, int]:
    options.validate()
    config_key = config_key_for_csv(csv_path)
    if config_key not in all_model_runs():
        raise ValueError(f"No Grad-CAM config for CSV: {csv_path}")

    df = pd.read_csv(csv_path, low_memory=False)
    image_column = infer_image_column(df, options.image_column)
    filtered = apply_filters(df, options)
    output_root = output_root_for_config(config_key, options)

    generated = 0
    skipped = 0
    failed = 0
    cam_methods = list(dict.fromkeys(options.cam_methods))
    cam_targets = list(dict.fromkeys(options.cam_targets))
    jobs: list[tuple[int, Path, dict[tuple[str, str], Path]]] = []
    progress = tqdm(
        filtered.iterrows(), total=len(filtered), desc=f"{config_key}: scan", unit="img"
    )
    for row_index, row in progress:
        image_path = valid_image(row[image_column])
        if image_path is None:
            failed += 1
            continue

        missing_outputs = [
            (method, target)
            for method in cam_methods
            for target in cam_targets
            if not options.only_missing
            or existing_gradcam_for_image(
                config_key, image_path, method=method, target=target
            )
            is None
        ]
        if not missing_outputs:
            skipped += 1
            continue

        if options.dry_run:
            generated += len(missing_outputs)
            continue

        jobs.append(
            (
                row_index,
                image_path,
                {
                    (method, target): output_path_for_image(
                        config_key, image_path, options, method=method, target=target
                    )
                    for method, target in missing_outputs
                },
            )
        )
    print(
        f"{csv_path.name}: scan queued={len(jobs)}, skipped={skipped}, failed={failed}"
    )

    if options.dry_run:
        print(
            f"{csv_path.name}: would_generate={generated}, skipped={skipped}, failed={failed}, output={output_root}"
        )
        return generated, skipped, failed

    if not jobs:
        print(
            f"{csv_path.name}: generated=0, skipped={skipped}, failed={failed}, output={output_root}"
        )
        return 0, skipped, failed

    from advanced_visualization.models.gradcam import load_gradcam_bundle

    bundle = load_gradcam_bundle(config_key)
    target_layer = bundle.target_layer
    device = bundle.device

    activations: dict = {}
    gradients: dict = {}

    def forward_hook(_module, _inputs, output):
        activations["value"] = output

    def backward_hook(_module, _grad_input, grad_output):
        gradients["value"] = grad_output[0]

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    dataset = GradcamDataset(
        jobs,
        bundle.transform,
        max_output_side=options.max_output_side,
    )
    loader = DataLoader(
        dataset,
        batch_size=options.batch_size,
        num_workers=options.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_valid,
        persistent_workers=options.num_workers > 0,
        prefetch_factor=options.prefetch_factor if options.num_workers > 0 else None,
        shuffle=False,
    )

    total_batches = int(np.ceil(len(jobs) / options.batch_size))
    progress = tqdm(
        loader, total=total_batches, desc=f"{csv_path.stem}: gradcam", unit="batch"
    )
    pending: list = []
    error_examples: list[str] = []
    max_pending_saves = max(options.save_workers * 4, options.batch_size * 2, 1)
    try:
        with ThreadPoolExecutor(max_workers=options.save_workers) as save_pool:
            for batch in progress:
                if batch is None:
                    continue
                try:
                    futures = process_batch(
                        config_key,
                        bundle,
                        batch,
                        activations,
                        gradients,
                        save_pool,
                        cam_methods,
                        cam_targets,
                    )
                    pending.extend(futures)
                except Exception as exc:
                    row_indices = batch[3]
                    failed += len(row_indices)
                    message = (
                        f"rows={row_indices[:5]} error={type(exc).__name__}: {exc}"
                    )
                    if len(error_examples) < options.max_error_examples:
                        error_examples.append(message)
                        progress.write(f"{csv_path.name}: failed batch {message}")
                    if options.stop_on_error:
                        raise RuntimeError(f"Failed batch: {exc}") from exc
                progress.set_postfix(
                    generated=generated,
                    skipped=skipped,
                    failed=failed,
                    queued=len(pending),
                )

                pending, saved, save_failed, save_errors = collect_finished_saves(
                    pending
                )
                generated += saved
                failed += save_failed
                for error in save_errors:
                    if len(error_examples) < options.max_error_examples:
                        error_examples.append(f"save error={error}")
                        progress.write(f"{csv_path.name}: failed save {error}")

                while len(pending) > max_pending_saves:
                    pending, saved, save_failed, save_errors = collect_finished_saves(
                        pending, block=True
                    )
                    generated += saved
                    failed += save_failed
                    for error in save_errors:
                        if len(error_examples) < options.max_error_examples:
                            error_examples.append(f"save error={error}")
                            progress.write(f"{csv_path.name}: failed save {error}")

            for fut in pending:
                ok, error = fut.result()
                if ok:
                    generated += 1
                else:
                    failed += 1
                    if error and len(error_examples) < options.max_error_examples:
                        error_examples.append(f"save error={error}")
    finally:
        forward_handle.remove()
        backward_handle.remove()

    if error_examples:
        print(f"{csv_path.name}: first {len(error_examples)} error example(s):")
        for error in error_examples:
            print(f"  - {error}")
    print(
        f"{csv_path.name}: generated={generated}, skipped={skipped}, failed={failed}, output={output_root}"
    )
    return generated, skipped, failed
