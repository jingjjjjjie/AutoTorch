"""Pre-generate Grad-CAM overlays for advanced_visualization CSVs."""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from advanced_visualization.core.artifacts import default_csv_paths as artifact_csv_paths
from advanced_visualization.core.artifacts import load_manifest
from advanced_visualization.core.config import DEFAULT_GRADCAM_ROOT, IMAGE_COLUMNS, all_model_runs, gradcam_artifact_root
from advanced_visualization.core.gradcam_cache import gradcam_cache_candidates, gradcam_roots
from advanced_visualization.models.gradcam import (
    compute_cam,
    gradcam_score,
    load_gradcam_bundle,
)
from advanced_visualization.core.images import valid_image
from advanced_visualization.core.settings import configured_path, load_settings


DEFAULT_IMAGE_COLUMNS = IMAGE_COLUMNS


def config_key_for_csv(csv_path: Path) -> str:
    manifest = load_manifest(csv_path.parent)
    if manifest and manifest.prepared_csv.resolve() == csv_path.resolve():
        return manifest.model_key
    for model in load_settings().models:
        if not model.enabled or not model.artifact_dir.strip():
            continue
        artifact_dir = configured_path(model.artifact_dir)
        if csv_path.resolve() == (artifact_dir / "prepared_predictions.csv").resolve() or csv_path.parent.resolve() == artifact_dir.resolve():
            return model.key
    return csv_path.stem


def default_csv_paths() -> list[Path]:
    model_runs = all_model_runs()
    return [path for path in artifact_csv_paths() if config_key_for_csv(path) in model_runs]


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
                paths.extend(sorted(candidate for candidate in path.glob("*.csv") if config_key_for_csv(candidate) in model_runs))
        else:
            paths.append(path)
    return paths


def infer_image_column(df: pd.DataFrame, requested: str | None) -> str:
    if requested:
        if requested not in df.columns:
            raise ValueError(f"Missing image column {requested!r}. Available columns: {list(df.columns)}")
        return requested
    for column in DEFAULT_IMAGE_COLUMNS:
        if column in df.columns:
            return column
    raise ValueError(f"No image path column found. Tried: {DEFAULT_IMAGE_COLUMNS}")


def apply_filters(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    filtered = df
    for raw_filter in args.filter:
        if "=" not in raw_filter:
            raise ValueError(f"Filter must be COLUMN=VALUE, got: {raw_filter}")
        column, value = raw_filter.split("=", 1)
        column = column.strip()
        value = value.strip()
        if column not in filtered.columns:
            raise ValueError(f"Filter column {column!r} not in CSV.")
        filtered = filtered[filtered[column].fillna("Missing").astype(str).eq(value)]

    if args.offset:
        filtered = filtered.iloc[args.offset :]
    if args.limit is not None:
        filtered = filtered.iloc[: args.limit]
    if args.num_shards > 1:
        filtered = filtered.iloc[args.shard_index :: args.num_shards]
    return filtered


def existing_gradcam_for_image(config_key: str, image_path: Path, method: str = "") -> Path | None:
    roots = gradcam_roots(config_key, "")
    for root in roots:
        for candidate in gradcam_cache_candidates(root, image_path, method=method):
            if candidate.is_file():
                return candidate
    return None


def output_root_for_config(config_key: str, args: argparse.Namespace) -> Path:
    if args.output_root:
        return args.output_root.expanduser() / config_key
    output_root = gradcam_artifact_root(config_key)
    if output_root is None:
        return DEFAULT_GRADCAM_ROOT / config_key
    return output_root


def output_path_for_image(config_key: str, image_path: Path, args: argparse.Namespace, method: str = "gradcam") -> Path:
    output_root = output_root_for_config(config_key, args)
    output_root.mkdir(parents=True, exist_ok=True)
    return gradcam_cache_candidates(output_root, image_path, method=method)[0]


class GradcamDataset(Dataset):
    def __init__(self, jobs: list[tuple[int, Path, dict[str, Path]]], transform):
        self.jobs = jobs
        self.transform = transform

    def __len__(self) -> int:
        return len(self.jobs)

    def __getitem__(self, index: int):
        row_index, image_path, output_paths = self.jobs[index]
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception:
            return None
        tensor = self.transform(image)
        base = np.asarray(image, dtype=np.uint8)
        return tensor, base, {method: str(path) for method, path in output_paths.items()}, row_index


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
    cam = np.nan_to_num(cam, nan=0.0, posinf=1.0, neginf=0.0)
    cam = np.clip(cam, 0.0, 1.0)
    base_f = base.astype(np.float32)
    heat = np.empty_like(base_f)
    heat[..., 0] = 255.0 * cam
    heat[..., 1] = 210.0 * np.sqrt(cam)
    heat[..., 2] = 28.0 * (1.0 - cam) * cam
    alpha = np.clip(0.18 + 0.55 * cam[..., None], 0.18, 0.65)
    overlay = base_f * (1.0 - alpha) + heat * alpha
    return Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))


def process_batch(config_key: str, bundle, batch, activations, gradients, save_pool: ThreadPoolExecutor, cam_methods: list[str]):
    tensors, bases, outputs, row_indices = batch
    model = bundle["model"]
    device = bundle["device"]

    activations.clear()
    gradients.clear()
    input_tensor = tensors.to(device, non_blocking=True)
    model.zero_grad(set_to_none=True)
    score = gradcam_score(model, input_tensor, config_key=config_key).sum()
    score.backward()

    if "value" not in activations:
        raise RuntimeError(f"Grad-CAM target layer did not produce activations for rows {row_indices[:5]}")
    if "value" not in gradients:
        raise RuntimeError(f"Grad-CAM target layer did not produce gradients for rows {row_indices[:5]}")

    activation = activations["value"].detach()
    gradient = gradients["value"].detach()
    futures = []
    for method in cam_methods:
        cam_batch = compute_cam(activation, gradient, config_key=config_key, method=method).float()
        for cam, base, output_paths in zip(cam_batch, bases, outputs):
            output_path = output_paths.get(method)
            if not output_path:
                continue
            h, w = base.shape[:2]
            resized = F.interpolate(cam.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False).squeeze()
            resized = (resized - resized.min()) / (resized.max() - resized.min() + 1e-8)
            cam_np = resized.detach().to("cpu", non_blocking=True).numpy()
            futures.append(save_pool.submit(_save_overlay, base, cam_np, output_path))
    return futures


def _save_overlay(base: np.ndarray, cam: np.ndarray, output_path: str) -> tuple[bool, str | None]:
    output = Path(output_path)
    tmp_output = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        overlay_from_cam(base, cam).save(tmp_output, format="PNG", compress_level=3)
        os.replace(tmp_output, output)
        return True, None
    except Exception as exc:
        tmp_output.unlink(missing_ok=True)
        return False, f"{type(exc).__name__}: {exc}"


def collect_finished_saves(pending: list, block: bool = False) -> tuple[list, int, int, list[str]]:
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


def pregenerate_csv(csv_path: Path, args: argparse.Namespace) -> tuple[int, int, int]:
    config_key = config_key_for_csv(csv_path)
    if config_key not in all_model_runs():
        raise ValueError(f"No Grad-CAM config for CSV: {csv_path}")

    df = pd.read_csv(csv_path, low_memory=False)
    image_column = infer_image_column(df, args.image_column)
    filtered = apply_filters(df, args)
    output_root = output_root_for_config(config_key, args)

    generated = 0
    skipped = 0
    failed = 0
    cam_methods = list(dict.fromkeys(args.cam_method))
    jobs: list[tuple[int, Path, dict[str, Path]]] = []
    progress = tqdm(filtered.iterrows(), total=len(filtered), desc=f"{config_key}: scan", unit="img")
    for row_index, row in progress:
        image_path = valid_image(row[image_column])
        if image_path is None:
            failed += 1
            continue

        missing_methods = [
            method
            for method in cam_methods
            if not args.only_missing or existing_gradcam_for_image(config_key, image_path, method=method) is None
        ]
        if not missing_methods:
            skipped += 1
            continue

        if args.dry_run:
            generated += len(missing_methods)
            continue

        jobs.append((row_index, image_path, {method: output_path_for_image(config_key, image_path, args, method=method) for method in missing_methods}))
    print(f"{csv_path.name}: scan queued={len(jobs)}, skipped={skipped}, failed={failed}")

    if args.dry_run:
        print(f"{csv_path.name}: would_generate={generated}, skipped={skipped}, failed={failed}, output={output_root}")
        return generated, skipped, failed

    if not jobs:
        print(f"{csv_path.name}: generated=0, skipped={skipped}, failed={failed}, output={output_root}")
        return 0, skipped, failed

    bundle = load_gradcam_bundle(config_key)
    target_layer = bundle["target_layer"]
    device = bundle["device"]

    activations: dict = {}
    gradients: dict = {}

    def forward_hook(_module, _inputs, output):
        activations["value"] = output

    def backward_hook(_module, _grad_input, grad_output):
        gradients["value"] = grad_output[0]

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    dataset = GradcamDataset(jobs, bundle["transform"])
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_valid,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        shuffle=False,
    )

    total_batches = int(np.ceil(len(jobs) / args.batch_size))
    progress = tqdm(loader, total=total_batches, desc=f"{csv_path.stem}: gradcam", unit="batch")
    pending: list = []
    error_examples: list[str] = []
    max_pending_saves = max(args.save_workers * 4, args.batch_size * 2, 1)
    try:
        with ThreadPoolExecutor(max_workers=args.save_workers) as save_pool:
            for batch in progress:
                if batch is None:
                    continue
                try:
                    futures = process_batch(config_key, bundle, batch, activations, gradients, save_pool, cam_methods)
                    pending.extend(futures)
                except Exception as exc:
                    row_indices = batch[3]
                    failed += len(row_indices)
                    message = f"rows={row_indices[:5]} error={type(exc).__name__}: {exc}"
                    if len(error_examples) < args.max_error_examples:
                        error_examples.append(message)
                        progress.write(f"{csv_path.name}: failed batch {message}")
                    if args.stop_on_error:
                        raise RuntimeError(f"Failed batch: {exc}") from exc
                progress.set_postfix(generated=generated, skipped=skipped, failed=failed, queued=len(pending))

                pending, saved, save_failed, save_errors = collect_finished_saves(pending)
                generated += saved
                failed += save_failed
                for error in save_errors:
                    if len(error_examples) < args.max_error_examples:
                        error_examples.append(f"save error={error}")
                        progress.write(f"{csv_path.name}: failed save {error}")

                while len(pending) > max_pending_saves:
                    pending, saved, save_failed, save_errors = collect_finished_saves(pending, block=True)
                    generated += saved
                    failed += save_failed
                    for error in save_errors:
                        if len(error_examples) < args.max_error_examples:
                            error_examples.append(f"save error={error}")
                            progress.write(f"{csv_path.name}: failed save {error}")

            for fut in pending:
                ok, error = fut.result()
                if ok:
                    generated += 1
                else:
                    failed += 1
                    if error and len(error_examples) < args.max_error_examples:
                        error_examples.append(f"save error={error}")
    finally:
        forward_handle.remove()
        backward_handle.remove()

    if error_examples:
        print(f"{csv_path.name}: first {len(error_examples)} error example(s):")
        for error in error_examples:
            print(f"  - {error}")
    print(f"{csv_path.name}: generated={generated}, skipped={skipped}, failed={failed}, output={output_root}")
    return generated, skipped, failed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        action="append",
        default=[],
        help="CSV file or artifact directory. Defaults to configured Settings artifacts.",
    )
    parser.add_argument("--image-column", default=None, help="Image path column. Defaults to absolute_ori_path if present.")
    parser.add_argument("--filter", action="append", default=[], help="Filter rows with COLUMN=VALUE. Can be repeated.")
    parser.add_argument("--offset", type=int, default=0, help="Skip this many filtered rows.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum rows per CSV.")
    parser.add_argument("--num-shards", type=int, default=1, help="Split filtered rows into this many shards.")
    parser.add_argument("--shard-index", type=int, default=0, help="Shard index for this process, from 0 to num-shards - 1.")
    parser.add_argument("--batch-size", type=int, default=32, help="Grad-CAM batch size.")
    parser.add_argument("--num-workers", type=int, default=min(8, os.cpu_count() or 1), help="DataLoader worker processes for image I/O and transforms.")
    parser.add_argument("--prefetch-factor", type=int, default=4, help="DataLoader prefetch factor (per worker).")
    parser.add_argument("--save-workers", type=int, default=8, help="Threads used to compose and save overlay PNGs.")
    parser.add_argument(
        "--cam-method",
        action="append",
        default=None,
        choices=["gradcam", "gradcam++"],
        help="CAM method to generate. Repeat for both gradcam and gradcam++.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Root directory for new overlays. Uses <output-root>/<csv-stem>. Existing prepared files are still detected in all viewer roots.",
    )
    parser.add_argument("--max-error-examples", type=int, default=10, help="Maximum batch/save errors to print per CSV.")
    parser.add_argument("--only-missing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true", help="Count work without generating files.")
    parser.add_argument("--stop-on-error", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_shards < 1:
        raise SystemExit("--num-shards must be >= 1")
    if not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("--shard-index must be in [0, num_shards)")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")
    if args.num_workers < 0:
        raise SystemExit("--num-workers must be >= 0")
    if args.save_workers < 1:
        raise SystemExit("--save-workers must be >= 1")
    if args.max_error_examples < 0:
        raise SystemExit("--max-error-examples must be >= 0")
    csv_paths = resolve_csv_paths(args.csv)
    if not csv_paths:
        raise SystemExit("No configured CSVs found.")
    if args.cam_method is None:
        args.cam_method = ["gradcam"]

    totals = [pregenerate_csv(path, args) for path in csv_paths]
    generated = sum(item[0] for item in totals)
    skipped = sum(item[1] for item in totals)
    failed = sum(item[2] for item in totals)
    print(f"Total: generated={generated}, skipped={skipped}, failed={failed}")


if __name__ == "__main__":
    main()
