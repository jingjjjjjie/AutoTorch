"""Merge model predictions and create stored PCA/t-SNE visualization materials."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


MODEL_SPECS = (
    (
        "realistic_internal_1024_square_ori",
        "Ex8point2res1024largerbs_square_exp_pred_ori",
        "pytorch",
        "ori",
        1024,
        "/mnt4/advanced_visualization/"
        "square_exp2_Ex8point2res1024_largerbs_21plusaugfeb_"
        "UniRepLKNet_T_legacy_v1_1024_ori_epoch8/checkpoints/epoch_8.pt",
    ),
    (
        "realistic_internal_1024_largerbs_ori",
        "Ex8point2res1024largerbs_pred_ori",
        "pytorch",
        "ori",
        1024,
        "/mnt4/advanced_visualization/"
        "Ex8point2res1024_moredata_largerbs_UniRepLKNet_T_legacy_v1_"
        "1024_ori_epoch11/checkpoints/epoch_11.pt",
    ),
    (
        "realistic_internal_ench21_vansmall_ori",
        "tf_ori_pred",
        "tensorflow",
        "ori",
        512,
        "/mnt3/auto-ekyc/idrecapture/artifacts/ori/"
        "Ench21_v1_20251006-1529/checkpoints/checkpoint_18.h5",
    ),
    (
        "realistic_internal_ench21_vansmall_crop",
        "tf_crop_pred",
        "tensorflow",
        "crop",
        512,
        "/mnt3/auto-ekyc/idrecapture/artifacts/crop/"
        "Ench21_v1_20251007-0040/checkpoints/checkpoint_15.h5",
    ),
    (
        "realistic_internal_ench21_overfitaffin_ori",
        "tf_overfit_affin_ori_pred",
        "tensorflow",
        "ori",
        512,
        "/mnt3/auto-ekyc/idrecapture/artifacts/ori/"
        "Ench21_v1_overfitAffin_20260722-1632/checkpoints/checkpoint_12.h5",
    ),
    (
        "realistic_internal_ench21_overfitaffin_crop",
        "tf_overfit_affin_crop_pred",
        "tensorflow",
        "crop",
        512,
        "/mnt3/auto-ekyc/idrecapture/artifacts/crop/"
        "Ench21_v1_overfitAffin_20260722-1657/checkpoints/checkpoint_19.h5",
    ),
)


ROUTER = '''"""Router for the realistic internal-testing feature workspace."""
from __future__ import annotations

import json
from pathlib import Path

ROUTER_API_VERSION = 1


def resolve(*, model_id: str, data_dir: Path) -> dict:
    manifest = json.loads(
        (data_dir / "visualization_manifest.json").read_text(encoding="utf-8")
    )
    layers = manifest.get("layers") or (manifest.get("gradcam") or {}).get("layers", [])
    prepared_layers = manifest.get("prepared_gradcam_layers")
    if prepared_layers is None:
        prepared_layers = [layer["key"] for layer in layers if layer.get("final")]
    return {
        "framework": manifest["framework"],
        "engine": manifest["engine"],
        "branch": manifest["branch"],
        "artifact_dir": data_dir,
        "checkpoint": manifest["checkpoint"],
        "prediction_data": data_dir / "features" / "prepared_predictions.csv",
        "feature_data": data_dir / "features" / "prepared_predictions.csv",
        "model_name": manifest["model_name"],
        "head_type": manifest["head_type"],
        "image_size": manifest["image_size"],
        "columns": manifest["columns"],
        "layers": layers,
        "prepared_gradcam_layers": prepared_layers,
    }
'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-source", required=True, type=Path)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--feature-input", action="append", default=[], type=Path)
    parser.add_argument(
        "--initialize-only",
        action="store_true",
        help="Create model routers/manifests before feature extraction.",
    )
    parser.add_argument(
        "--truth-label",
        type=int,
        choices=(0, 1),
        help="Normalize the truth label stored with every model feature row.",
    )
    return parser.parse_args()


def stored_projections(frame: pd.DataFrame, prediction_column: str) -> pd.DataFrame:
    feature_columns = [
        column for column in frame.columns if str(column).startswith("feature_")
    ]
    if not feature_columns:
        raise ValueError(f"No feature columns found for {prediction_column}.")
    matrix = StandardScaler().fit_transform(
        frame[feature_columns].to_numpy(dtype="float32")
    )
    pca = PCA(n_components=2, random_state=42).fit_transform(matrix)
    perplexity = min(30, max(2, len(frame) - 1))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=42,
    ).fit_transform(matrix)
    return pd.DataFrame(
        {
            "uuid": frame["uuid"].astype(str),
            prediction_column: frame[prediction_column],
            "pca_x": pca[:, 0],
            "pca_y": pca[:, 1],
            "tsne_x": tsne[:, 0],
            "tsne_y": tsne[:, 1],
        }
    )


def complete_prediction_table(
    merged: pd.DataFrame,
    source_columns: list[str],
    prediction_columns: list[str],
) -> pd.DataFrame:
    """Return the canonical dataset rows scored by every registered model."""

    missing_columns = [
        column
        for column in [*source_columns, *prediction_columns]
        if column not in merged.columns
    ]
    if missing_columns:
        raise ValueError(
            "Cannot build unified predictions; missing columns: "
            + ", ".join(missing_columns)
        )
    output_columns = list(dict.fromkeys([*source_columns, *prediction_columns]))
    complete = merged[prediction_columns].notna().all(axis=1)
    return merged.loc[complete, output_columns].reset_index(drop=True)


def write_model_contract(workspace: Path, spec: tuple) -> None:
    model_id, prediction_column, framework, branch, image_size, checkpoint = spec
    model_dir = workspace / model_id
    model_dir.mkdir(parents=True, exist_ok=True)
    image_column = "absolute_crop_path" if branch == "crop" else "absolute_ori_path"
    if framework == "tensorflow":
        layers = [
            {"key": "norm3", "label": "norm3", "module": "norm3", "max_long_edge": 768},
            {"key": "block3_3", "label": "block3.3", "module": "block3.3", "max_long_edge": 768},
            {"key": "block4_1", "label": "block4.1", "module": "block4.1", "max_long_edge": 768},
            {"key": "norm4", "label": "norm4 (final)", "module": "norm4", "final": True, "max_long_edge": 1024},
        ]
        prepared_gradcam_layers = ["norm4"]
    else:
        layers = [
            {"key": "stage1", "label": "Stage 1", "module": "feature_extractor.stages.0", "max_long_edge": 768},
            {"key": "stage2", "label": "Stage 2", "module": "feature_extractor.stages.1", "max_long_edge": 768},
            {"key": "stage3", "label": "Stage 3", "module": "feature_extractor.stages.2", "max_long_edge": 768},
            {"key": "stage4", "label": "Stage 4 (final)", "module": "feature_extractor.stages.3", "final": True, "max_long_edge": 1024},
        ]
        prepared_gradcam_layers = ["stage4"]
    manifest = {
        "schema_version": 1,
        "dataset": "realistic_internal_testing",
        "model_id": model_id,
        "framework": framework,
        "engine": "vansmall" if framework == "tensorflow" else "unireplknet",
        "branch": branch,
        "checkpoint": checkpoint,
        "prepared_predictions": "features/prepared_predictions.csv",
        "stored_projections": "features/stored_projections.csv",
        "model_name": "vansmall" if framework == "tensorflow" else "unireplknet_t",
        "head_type": "tensorflow_export" if framework == "tensorflow" else "legacy_v1",
        "image_size": image_size,
        "columns": {
            "sample_id": "uuid",
            "image": image_column,
            "truth": "label",
            "prediction": prediction_column,
            "subclass": "Recapture_Subclass",
        },
        "layers": layers,
        "prepared_gradcam_layers": prepared_gradcam_layers,
    }
    (model_dir / "visualization_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (model_dir / "router.py").write_text(ROUTER, encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.workspace.mkdir(parents=True, exist_ok=True)
    if args.initialize_only:
        for spec in MODEL_SPECS:
            write_model_contract(args.workspace, spec)
        print(f"Initialized {len(MODEL_SPECS)} model contracts in {args.workspace}")
        return
    if len(args.feature_input) != len(MODEL_SPECS):
        raise ValueError(f"Expected {len(MODEL_SPECS)} --feature-input values.")
    source = pd.read_csv(args.prepared_source, low_memory=False)
    merged = source.copy()

    for spec, feature_input in zip(MODEL_SPECS, args.feature_input):
        model_id, prediction_column, framework, branch, image_size, checkpoint = spec
        features = pd.read_csv(feature_input, low_memory=False)
        if args.truth_label is not None:
            features["label"] = args.truth_label
        if features["uuid"].isna().any() or features["uuid"].astype(str).duplicated().any():
            raise ValueError(f"{model_id}: feature uuid must be populated and unique.")
        unknown = set(features["uuid"].astype(str)) - set(source["uuid"].astype(str))
        if unknown:
            raise ValueError(f"{model_id}: feature CSV contains unknown uuid values.")
        if prediction_column not in features.columns:
            raise ValueError(f"{model_id}: missing {prediction_column}.")
        if features[prediction_column].isna().any():
            raise ValueError(f"{model_id}: prediction contains missing values.")

        model_dir = args.workspace / model_id
        feature_dir = model_dir / "features"
        feature_dir.mkdir(parents=True, exist_ok=True)
        prepared_output = feature_dir / "prepared_predictions.csv"
        features.to_csv(prepared_output, index=False)
        stored_projections(features, prediction_column).to_csv(
            feature_dir / "stored_projections.csv", index=False
        )

        write_model_contract(args.workspace, spec)
        prediction_map = features.set_index(features["uuid"].astype(str))[
            prediction_column
        ]
        merged[prediction_column] = merged["uuid"].astype(str).map(prediction_map)

    original_columns = pd.read_csv(
        args.prepared_source, nrows=0
    ).columns.tolist()
    prediction_columns = [spec[1] for spec in MODEL_SPECS]
    source_columns = [
        column
        for column in original_columns
        if column not in prediction_columns
    ]
    complete = complete_prediction_table(merged, source_columns, prediction_columns)
    merged_output = args.workspace / "index_annotation_with_predictions.csv"
    complete.to_csv(merged_output, index=False)
    print(
        f"Wrote {len(complete)} complete prediction rows to {merged_output} "
        f"(dropped {len(merged) - len(complete)} rows missing at least one prediction)"
    )


if __name__ == "__main__":
    main()
