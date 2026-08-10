"""Generate one router-registered PyTorch Grad-CAM++ layer in correct batches."""

from __future__ import annotations

import argparse
import gc
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from tqdm import tqdm

from advanced_visualization.core.images import valid_image
from advanced_visualization.core.model_router import registered_model_routes
from advanced_visualization.core.registered_gradcam import (
    TARGETS,
    _atomic_json,
    _base_image,
    _model_config,
    _module_at_path,
    _output_path,
    _save_webp,
    initialize_artifact,
)
from advanced_visualization.models.registry import get_gradcam_engine


def generate(
    model_id: str,
    layer_key: str,
    *,
    batch_size: int,
    num_shards: int,
    shard_index: int,
    overwrite: bool,
    artifact_root: Path | None = None,
) -> dict:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if num_shards < 1:
        raise ValueError("num_shards must be at least 1.")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("shard_index must be in [0, num_shards).")

    route = registered_model_routes()[model_id]
    if route.framework != "pytorch":
        raise ValueError(f"{model_id} is not a PyTorch route.")
    layer = route.layer(layer_key)
    image_column = route.columns["image"]
    artifact_dir = initialize_artifact(route, artifact_root)

    engine = get_gradcam_engine(route.engine)
    bundle = engine.load_bundle(_model_config(route))
    model = bundle.model.eval()
    model.requires_grad_(False)
    target_layer = _module_at_path(model, layer.module)

    frame = pd.read_csv(route.prediction_data, usecols=[image_column], low_memory=False)
    image_paths = []
    seen = set()
    invalid_sources = 0
    for raw_path in frame[image_column]:
        image_path = valid_image(raw_path)
        if image_path is None:
            invalid_sources += 1
            continue
        resolved = str(image_path.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        image_paths.append(image_path)
    image_paths = image_paths[shard_index::num_shards]

    state_path = artifact_dir / (
        f"gradcam_generation_state.{layer.key}."
        f"shard-{shard_index:03d}-of-{num_shards:03d}.json"
    )
    generated = skipped = failed = oom_splits = processed = 0
    safe_compute_batch_size = batch_size
    save_workers = max(1, int(os.environ.get("AUTOTORCH_GRADCAM_SAVE_WORKERS", "14")))
    save_pool = ThreadPoolExecutor(
        max_workers=save_workers,
        thread_name_prefix="gradcam-webp",
    )
    pending_save_jobs = []
    progress = tqdm(
        total=len(image_paths),
        desc=f"{model_id}:{layer.key}:shard-{shard_index:03d}:batch-{batch_size}",
        unit="img",
    )

    def settle_save_job(job) -> None:
        nonlocal generated, failed
        image_path, target, future = job
        try:
            future.result()
            generated += 1
        except Exception as exc:
            failed += 1
            progress.write(
                f"{model_id}:{layer.key}: {image_path}: {target}: "
                f"{type(exc).__name__}: {exc}"
            )

    def collect_save_jobs(*, wait: bool) -> None:
        remaining = []
        for job in pending_save_jobs:
            if wait or job[2].done():
                settle_save_job(job)
            else:
                remaining.append(job)
        pending_save_jobs[:] = remaining

    def compute(items):
        nonlocal generated, failed, oom_splits, safe_compute_batch_size
        if not items:
            return None
        if len(items) > safe_compute_batch_size:
            return tuple(
                items[start : start + safe_compute_batch_size]
                for start in range(0, len(items), safe_compute_batch_size)
            )
        try:
            stacked = torch.stack([item[2] for item in items])
            cams = {}
            needed_targets = [
                target
                for target in TARGETS
                if any(target in item[4] for item in items)
            ]
            captured = {}
            inputs = scores = activation = gradient = None

            def forward_hook(_module, _inputs, output):
                captured["activation"] = output

            forward_handle = target_layer.register_forward_hook(forward_hook)
            try:
                inputs = stacked.to(bundle.device).requires_grad_(True)
                model.zero_grad(set_to_none=True)
                with torch.enable_grad():
                    scores = engine.score(model, inputs)
                    activation = captured["activation"]
                    gradient = torch.autograd.grad(scores.sum(), activation)[0]
                detached_activation = activation.detach()
                detached_gradient = gradient.detach()
                for target in needed_targets:
                    target_gradient = (
                        -detached_gradient if target == "genuine" else detached_gradient
                    )
                    cams[target] = engine.compute_cam(
                        detached_activation,
                        target_gradient,
                        method="gradcam++",
                    ).float().cpu()
            finally:
                forward_handle.remove()
                captured.clear()
                inputs = scores = activation = gradient = None

            save_jobs = []
            for item_index, (image_path, base, _tensor, outputs, missing) in enumerate(items):
                for target in missing:
                    try:
                        cam = cams[target][item_index : item_index + 1]
                        resized = F.interpolate(
                            cam,
                            size=base.shape[:2],
                            mode="bilinear",
                            align_corners=False,
                        ).squeeze()
                        resized = (resized - resized.min()) / (
                            resized.max() - resized.min() + 1e-8
                        )
                        save_jobs.append(
                            (
                                image_path,
                                target,
                                save_pool.submit(
                                    _save_webp,
                                    base,
                                    resized.numpy().copy(),
                                    outputs[target],
                                ),
                            )
                        )
                    except Exception as exc:
                        failed += 1
                        progress.write(
                            f"{model_id}:{layer.key}: {image_path}: {target}: "
                            f"{type(exc).__name__}: {exc}"
                        )
            pending_save_jobs.extend(save_jobs)
            collect_save_jobs(wait=False)
            max_pending_saves = save_workers * 4
            while len(pending_save_jobs) > max_pending_saves:
                settle_save_job(pending_save_jobs.pop(0))
        except torch.OutOfMemoryError:
            if len(items) == 1:
                failed += len(items[0][4])
                progress.write(
                    f"{model_id}:{layer.key}: {items[0][0]}: CUDA out of memory "
                    "at batch size 1"
                )
                return ()
            oom_splits += 1
            reduced_size = len(items) - 1 if len(items) > 2 else 1
            safe_compute_batch_size = min(safe_compute_batch_size, reduced_size)
            return (items[:reduced_size], items[reduced_size:])

    next_state_at = 100
    for start in range(0, len(image_paths), batch_size):
        prepared = []
        batch_paths = image_paths[start : start + batch_size]
        for image_path in batch_paths:
            outputs = {
                target: _output_path(artifact_dir, image_path, layer.key, target)
                for target in TARGETS
            }
            missing = [
                target
                for target, output in outputs.items()
                if overwrite or not output.is_file()
            ]
            if not missing:
                skipped += len(TARGETS)
                continue
            try:
                with Image.open(image_path) as opened:
                    image = ImageOps.exif_transpose(opened).convert("RGB")
                    base = _base_image(image, layer.max_long_edge)
                    transformed = bundle.transform(image)
                prepared.append((image_path, base, transformed, outputs, missing))
            except Exception as exc:
                failed += len(missing)
                progress.write(
                    f"{model_id}:{layer.key}: {image_path}: "
                    f"{type(exc).__name__}: {exc}"
                )
        pending = [prepared]
        while pending:
            split = compute(pending.pop())
            if split is not None:
                gc.collect()
                if bundle.device.type == "cuda":
                    torch.cuda.empty_cache()
                pending.extend(reversed(split))
        processed += len(batch_paths)
        progress.update(len(batch_paths))
        progress.set_postfix(
            generated=generated,
            skipped=skipped,
            failed=failed,
            oom_splits=oom_splits,
        )
        if processed >= next_state_at:
            peak_mb = round(torch.cuda.max_memory_reserved(0) / 1024 / 1024, 1)
            _atomic_json(
                state_path,
                {
                    "model_id": model_id,
                    "layer": layer.key,
                    "batch_size": batch_size,
                    "num_shards": num_shards,
                    "shard_index": shard_index,
                    "source_images": len(image_paths),
                    "processed": processed,
                    "generated": generated,
                    "skipped": skipped,
                    "failed": failed,
                    "oom_splits": oom_splits,
                    "effective_batch_size": safe_compute_batch_size,
                    "invalid_sources": invalid_sources,
                    "torch_peak_reserved_mib": peak_mb,
                    "complete": False,
                },
            )
            while next_state_at <= processed:
                next_state_at += 100

    collect_save_jobs(wait=True)
    progress.set_postfix(
        generated=generated,
        skipped=skipped,
        failed=failed,
        oom_splits=oom_splits,
    )
    progress.close()
    save_pool.shutdown()
    peak_mb = round(torch.cuda.max_memory_reserved(0) / 1024 / 1024, 1)
    result = {
        "model_id": model_id,
        "layer": layer.key,
        "batch_size": batch_size,
        "num_shards": num_shards,
        "shard_index": shard_index,
        "source_images": len(image_paths),
        "processed": processed,
        "generated": generated,
        "skipped": skipped,
        "failed": failed,
        "oom_splits": oom_splits,
        "effective_batch_size": safe_compute_batch_size,
        "invalid_sources": invalid_sources,
        "torch_peak_reserved_mib": peak_mb,
        "complete": True,
    }
    _atomic_json(state_path, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_id")
    parser.add_argument("--layer", required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--artifact-root", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    print(
        json.dumps(
            generate(
                arguments.model_id,
                arguments.layer,
                batch_size=arguments.batch_size,
                num_shards=arguments.num_shards,
                shard_index=arguments.shard_index,
                overwrite=arguments.overwrite,
                artifact_root=arguments.artifact_root,
            ),
            sort_keys=True,
        )
    )
