# IDRecapture TensorFlow Artifact Export Request

## What We Need

Build an exporter in the TensorFlow IDRecapture repo that prepares artifacts for
AutoTorch advanced visualization.

AutoTorch will not load TensorFlow/Keras models. It will only read:

- a prepared CSV;
- resolved image paths;
- precomputed VAN Small 2x2 Grad-CAM montage PNGs;
- optional numeric feature columns.

## Target TensorFlow Repo

```text
/home/jingjie/Dev/automl_platform/idrecapture_server
```

Suggested new script:

```text
idfraud/export_autotorch_viz_artifact.py
```

Do not change training or evaluation behavior unless strictly necessary. This
should be an export/adapter layer.

## Target AutoTorch Output Root

Save exported visualization artifacts under `/mnt5`, matching the current
AutoTorch visualization artifact convention:

```text
/mnt5/temp_jj/<tf_vansmall_run_name>/
```

The final folder must look like:

```text
/mnt5/temp_jj/<tf_vansmall_run_name>/
  prepared_predictions.csv
  visualization_manifest.json
  montage/
    crop/
      <uuid>_crop_layer_montage.png
    ori/
      <uuid>_ori_layer_montage.png
```

## Source Model Context

Primary Ench21 parallel model:

```text
/mnt3/auto-ekyc/idrecapture/artifacts/parallel/Ench21_v1_20251010-1106/parallel_Ench21_v1_20251007-0040_Ench21_v1_20251006-1529.h5
```

Model artifact directory:

```text
/mnt3/auto-ekyc/idrecapture/artifacts/parallel/Ench21_v1_20251010-1106
```

This is a parallel merged Keras model built from:

- crop model: `Ench21_v1_20251007-0040`
- crop best checkpoint:
  `/mnt3/auto-ekyc/idrecapture/artifacts/crop/Ench21_v1_20251007-0040/checkpoints/checkpoint_15.h5`
- ori model: `Ench21_v1_20251006-1529`
- ori best checkpoint:
  `/mnt3/auto-ekyc/idrecapture/artifacts/ori/Ench21_v1_20251006-1529/checkpoints/checkpoint_18.h5`
- merge type: `average`
- TensorFlow version recorded in logs: `2.8.0`

## Existing IDRecapture Inputs

The exporter should consume existing evaluation and Grad-CAM outputs when
available:

```text
<model_dir>/infer_results/<dataset>/<dataset>.csv
<model_dir>/infer_results/<dataset>/gradcam/heatmap_<dataset>.csv
<model_dir>/infer_results/<dataset>/gradcam/ori_heatmap_npy/*.npy
<model_dir>/infer_results/<dataset>/gradcam/crop_heatmap_npy/*.npy
```

Where:

```text
<model_dir> = /mnt3/auto-ekyc/idrecapture/artifacts/parallel/Ench21_v1_20251010-1106
```

## Target / Alignment CSV From AutoTorch

The exported artifact should be compatible with the current AutoTorch review
CSV:

```text
/home/jingjie/AutoTorch/src/eval/idfraud/annotation/joined_predictions.csv
```

Use this CSV as the target review schema when possible. In particular, preserve
or align these common columns if they are available:

```text
uuid
label
absolute_ori_path
absolute_ocr_path
Recapture_Subclass
Quality_Issue
Data_Identity
```

The exporter may use the IDRecapture evaluation CSV as its prediction source,
but the final `prepared_predictions.csv` should be easy to join back to
`joined_predictions.csv`, preferably by `uuid`.

If both sources are provided:

```text
--idrecapture-result-csv <model_dir>/infer_results/<dataset>/<dataset>.csv
--target-review-csv /home/jingjie/AutoTorch/src/eval/idfraud/annotation/joined_predictions.csv
```

then output one row per target review CSV row where possible, with TensorFlow
prediction and Grad-CAM columns added.

If existing `.npy` heatmaps are missing or not compatible with Ench21, generate
new Grad-CAMs inside the TensorFlow repo.

For Ench21 VAN Small, compute the requested Grad-CAM layers with pre-sigmoid
logit as the target score. Do not use the old ResNet-specific
`conv5_block3_out`.

## Required CSV: prepared_predictions.csv

The exporter must write:

```text
/mnt5/temp_jj/<tf_vansmall_run_name>/prepared_predictions.csv
```

Required columns:

```text
uuid
label
absolute_ori_path
absolute_ocr_path
tf_parallel_pred
tf_crop_pred
tf_ori_pred
tf_crop_layer_montage_path
tf_ori_layer_montage_path
```

Map prediction columns from the existing IDRecapture result CSV:

```text
tf_parallel_pred <- ypred_raw
tf_crop_pred     <- crop_ypred_raw
tf_ori_pred      <- ori_ypred_raw
```

Preserve useful original metadata columns, for example:

```text
fraud_type
batch_name
case
ori_path
ocr_path
ypred
```

## Image Path Resolution

The `absolute_ori_path` and `absolute_ocr_path` columns must be valid absolute
paths to readable image files.

Resolve paths using the same fallback behavior as `EvaluateModel`:

1. `root_path.mnt_path`
2. `root_path.primary_dataset_path`
3. `root_path.secondary_dataset_path/image_source`
4. `root_path.live_path`

Use the appropriate source columns:

```text
absolute_ori_path <- resolved ori_path
absolute_ocr_path <- resolved ocr_path
```

## Grad-CAM PNG Requirement

AutoTorch visualization wants viewable PNG overlays, not raw `.npy` heatmaps.

For VAN Small, use pre-sigmoid logit as the Grad-CAM score. Do not use
post-sigmoid probability for Grad-CAM because Ench21 VAN Small predictions can
saturate at probability `1.0`, making sigmoid gradients uninformative.

Production default:

```text
score: pre-sigmoid logit
main layer: norm3
```

Optional expert comparison layers:

```text
block3.3
block4.1
norm4
```

For the current VAN Small delivery, keep montage artifacts only. The exporter
should still compute each requested layer internally, then compose a 2x2 montage
per branch/sample and delete or avoid preserving intermediate per-layer PNGs.

```text
/mnt5/temp_jj/<tf_vansmall_run_name>/montage/crop/<uuid>_crop_layer_montage.png
/mnt5/temp_jj/<tf_vansmall_run_name>/montage/ori/<uuid>_ori_layer_montage.png
```

Write absolute paths to the montage PNGs into:

```text
tf_crop_layer_montage_path
tf_ori_layer_montage_path
```

The AutoTorch VAN Small workspace displays these montage PNGs directly. The
general single-layer Grad-CAM path columns are not required for this delivery.

The exporter should treat Crop and Ori as separate branches:

```text
Crop:
input column: absolute_ocr_path
model weight: crop checkpoint
output column: tf_crop_layer_montage_path

Ori:
input column: absolute_ori_path
model weight: ori checkpoint
output column: tf_ori_layer_montage_path
```

Required for the VAN Small workspace:

```text
tf_crop_layer_montage_path
tf_ori_layer_montage_path
```

Per-layer Grad-CAM path columns are no longer required. If present, they can be
empty. The internal layer order for each 2x2 montage should be:

```text
norm3
block3_3
block4_1
norm4
```

Still use the pre-sigmoid logit as the Grad-CAM score, not post-sigmoid
probability.

Expected branch commands:

```bash
# Crop branch
--branches crop --montage-only --skip-features --gpu 0

# Ori branch
--branches ori --montage-only --skip-features --gpu 1
```

## Optional Feature Columns

If practical, export embeddings/features for feature-space visualization.

Add numeric columns named:

```text
feature_0000
feature_0001
feature_0002
...
```

Recommended feature source for Ench21 VAN Small:

- `avg_pool`; or
- pooled `norm4`.

For the parallel model, acceptable first implementation:

1. extract crop features;
2. extract ori features;
3. concatenate them into one numeric vector;
4. write the concatenated vector as `feature_0000...feature_N`.

If feature export is too large for the first pass, skip it and still deliver the
required CSV plus Grad-CAM PNG paths.

## Manifest

Write:

```text
/mnt5/temp_jj/<tf_vansmall_run_name>/visualization_manifest.json
```

Use this schema:

```json
{
  "artifact_dir": "/mnt5/temp_jj/<tf_vansmall_run_name>",
  "prepared_csv": "/mnt5/temp_jj/<tf_vansmall_run_name>/prepared_predictions.csv",
  "source_csv": "/mnt3/auto-ekyc/idrecapture/artifacts/parallel/Ench21_v1_20251010-1106/infer_results/<dataset>/<dataset>.csv",
  "model_key": "<tf_vansmall_run_name>",
  "checkpoint": "/mnt3/auto-ekyc/idrecapture/artifacts/parallel/Ench21_v1_20251010-1106/parallel_Ench21_v1_20251007-0040_Ench21_v1_20251006-1529.h5",
  "gradcam_dir": "/mnt5/temp_jj/<tf_vansmall_run_name>/montage",
  "image_column": "absolute_ori_path",
  "item_id_column": "uuid",
  "truth_column": "label",
  "prediction_column": "tf_parallel_pred",
  "subclass_column": "fraud_type"
}
```

If `fraud_type` is unavailable, use another useful metadata column or leave
`subclass_column` empty.

## AutoTorch Settings Entry

After export, AutoTorch can consume the folder with this settings entry:

```json
{
  "key": "<tf_vansmall_run_name>",
  "prediction_csv": "/mnt5/temp_jj/<tf_vansmall_run_name>/prepared_predictions.csv",
  "feature_csv": "/mnt5/temp_jj/<tf_vansmall_run_name>/prepared_predictions.csv",
  "artifact_dir": "/mnt5/temp_jj/<tf_vansmall_run_name>",
  "checkpoint": "",
  "weights_epoch": null,
  "model_type": "artifact_only",
  "model_name": "external_tensorflow_ench21",
  "head_type": "",
  "image_size": 512,
  "image_column": "absolute_ori_path",
  "prediction_column": "tf_parallel_pred",
  "enabled": true
}
```

Important:

```text
model_type must be artifact_only.
```

AutoTorch will not load TensorFlow weights.

## Success Criteria

The task is done when:

1. `/mnt5/temp_jj/<tf_vansmall_run_name>/prepared_predictions.csv` exists.
2. The CSV has valid absolute image paths in `absolute_ori_path` and/or `absolute_ocr_path`.
3. The CSV has numeric `tf_parallel_pred`.
4. The CSV has `label`.
5. The CSV has `tf_crop_layer_montage_path` and/or `tf_ori_layer_montage_path`.
6. Montage path columns point to readable PNG files.
7. Optional: the CSV has `feature_0000...` columns for feature-space visualization.
8. AutoTorch can open the exported artifact via:

```bash
streamlit run /home/jingjie/AutoTorch/advanced_visualization/unified_app.py
```

## Summary

Please build an IDRecapture TensorFlow-side exporter that turns existing Ench21
evaluation outputs, Grad-CAM heatmaps, and optional features into an AutoTorch
artifact-only visualization folder under `/mnt5/temp_jj`. AutoTorch should only
read the exported files; TensorFlow model loading remains entirely in
`idrecapture_server`.
