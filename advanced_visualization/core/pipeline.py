"""Single unified preparation pipeline for visualization artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from advanced_visualization.core.artifacts import load_manifest
from advanced_visualization.core.config import all_model_runs
from advanced_visualization.core.gradcam_generation import (
    GradcamGenerationOptions,
    pregenerate_csv,
)
from advanced_visualization.core.preparation import prepare_artifact
from advanced_visualization.core.settings import load_settings


@dataclass(frozen=True)
class PipelineResult:
    model_key: str
    prepared_csv: Path
    generated_gradcam: int
    skipped_gradcam: int
    failed_gradcam: int


@dataclass(frozen=True)
class PipelineOptions:
    batch_size: int
    gradcam_batch_size: int
    num_workers: int
    save_workers: int
    only_missing_gradcam: bool
    generate_gradcam: bool
    extract_features: bool
    cam_methods: tuple[str, ...] = ("gradcam",)
    limit: int | None = None


def pipeline_options_from_settings() -> PipelineOptions:
    pipeline = load_settings().pipeline
    return PipelineOptions(
        batch_size=int(pipeline["batch_size"]),
        gradcam_batch_size=int(pipeline["gradcam_batch_size"]),
        num_workers=int(pipeline["num_workers"]),
        save_workers=int(pipeline["save_workers"]),
        only_missing_gradcam=bool(pipeline["only_missing_gradcam"]),
        generate_gradcam=bool(pipeline["generate_gradcam"]),
        extract_features=bool(pipeline["extract_features"]),
        cam_methods=tuple(pipeline.get("cam_methods", ["gradcam"])),
        limit=int(pipeline["gradcam_limit"]) or None,
    )


def prepare_model_artifact(
    *,
    model_key: str,
    artifact_dir: Path,
    prediction_csv: Path,
    weights_epoch: int,
    options: PipelineOptions | None = None,
) -> PipelineResult:
    if options is None:
        options = pipeline_options_from_settings()
    model_runs = all_model_runs()
    if model_key not in model_runs:
        raise ValueError(f"Unknown model key: {model_key}")
    config = model_runs[model_key]

    prepared_csv = prepare_artifact(
        artifact_dir=artifact_dir,
        pred_csv=prediction_csv,
        epoch=weights_epoch,
        model_key=model_key,
    )

    manifest = load_manifest(artifact_dir)
    image_column = manifest.image_column if manifest else ""
    if options.extract_features:
        if not image_column:
            raise ValueError(
                f"Cannot extract features for {model_key}: no image column inferred."
            )
        from advanced_visualization.core.feature_extraction import (
            extract_features_and_predictions,
        )

        extract_features_and_predictions(
            config=config,
            csv_path=prepared_csv,
            image_column=image_column,
            output_csv=prepared_csv,
            batch_size=options.batch_size,
            num_workers=options.num_workers,
        )

    generated = skipped = failed = 0
    if options.generate_gradcam:
        generated, skipped, failed = pregenerate_csv(
            prepared_csv,
            GradcamGenerationOptions(
                image_column=image_column or None,
                batch_size=options.gradcam_batch_size,
                num_workers=options.num_workers,
                save_workers=options.save_workers,
                only_missing=options.only_missing_gradcam,
                cam_methods=options.cam_methods,
                cam_targets=("fraud",),
                limit=options.limit,
            ),
        )

    return PipelineResult(
        model_key=model_key,
        prepared_csv=prepared_csv,
        generated_gradcam=generated,
        skipped_gradcam=skipped,
        failed_gradcam=failed,
    )
