# Adding Models to Advanced Visualization

This guide explains what must exist before a model can run through the
advanced visualization preparation pipeline.

## Current Support

The current pipeline supports PyTorch models that can be loaded by AutoTorch.
The Streamlit viewer itself is artifact-first: it reads prepared CSVs, feature
columns, manifests, and Grad-CAM image files.

It does not currently support TensorFlow or Keras models. TensorFlow support
would need a new model adapter layer for loading, preprocessing, inference,
feature extraction, and saliency/Grad-CAM generation.

## Pipeline Expectations

`python advanced_visualization/cli/prepare_all.py` expects each enabled
model-loading model in `advanced_visualization/settings.json` to provide:

- `key`: unique run name.
- `prediction_csv`: source CSV with image paths and labels/predictions.
- `artifact_dir`: directory where prepared artifacts are written.
- `checkpoint` or `weights_epoch`: PyTorch checkpoint path.
- `model_type`: visualization engine name, for example `unireplknet`.
- `model_name`: AutoTorch backbone name, for example `unireplknet_t`.
- `head_type`: AutoTorch classification head name.
- `image_size`: transform input size.
- `image_column`: CSV column containing image paths.
- `prediction_column`: output column to write model predictions into.

The pipeline then:

1. creates a prepared CSV and manifest;
2. loads the model bundle for the configured `model_type`;
3. extracts feature columns named `feature_0000`, `feature_0001`, etc.;
4. writes predictions into `prediction_column`;
5. generates prepared Grad-CAM overlays.

Models with `"model_type": "artifact_only"` are skipped by `prepare_all.py`.
They are viewer-only entries for artifacts generated outside AutoTorch.

## Case 1: New Run of an Existing Supported Model

If the new model is still a supported AutoTorch UniRepLKNet classifier, no new
Python code is required.

Add a model entry in the Settings page or edit
`advanced_visualization/settings.json`:

```json
{
  "key": "my_unireplknet_run_epoch12",
  "prediction_csv": "/path/to/predictions.csv",
  "feature_csv": "",
  "artifact_dir": "/path/to/run_dir",
  "checkpoint": "/path/to/run_dir/checkpoints/epoch_12.pt",
  "weights_epoch": 12,
  "model_type": "unireplknet",
  "model_name": "unireplknet_t",
  "head_type": "legacy_v1",
  "image_size": 512,
  "image_column": "absolute_ori_path",
  "prediction_column": "my_unireplknet_pred",
  "enabled": true
}
```

Run:

```bash
python advanced_visualization/cli/prepare_all.py --model-key my_unireplknet_run_epoch12
```

## Case 2: New PyTorch Backbone in AutoTorch

If the architecture is still a PyTorch AutoTorch classifier but uses a new
backbone, implement the backbone in `src/models/backbones/`.

Required changes:

- Add a loader module, for example `src/models/backbones/my_backbone.py`.
- The loader must return `(model, output_dim)`.
- Register model names in `src/models/backbones/__init__.py` by adding them to
  `BACKBONE_LOADERS`.
- Make sure the resulting full model still exposes:
  - `model.feature_extractor(input_tensor)` for embeddings;
  - `model.mlp_head(features)` for prediction.

If those attributes remain compatible, the existing `unireplknet` visualization
engine may work only when its target-layer logic is also compatible. If the
target layer differs, create a separate visualization engine as described in
Case 3.

## Case 3: New Visualization Model Family

Create a new model-specific engine when the new model needs different loading,
target-layer selection, score logic, CAM computation, or feature extraction.

Add:

```text
advanced_visualization/models/<model_type>/__init__.py
advanced_visualization/models/<model_type>/gradcam.py
```

Implement the `GradcamEngine` protocol from
`advanced_visualization/models/base.py`:

```python
class MyModelGradcamEngine:
    name = "my_model_type"

    def load_bundle(self, config):
        ...

    def score(self, model, input_tensor):
        ...

    def compute_cam(self, activation, gradient, method="gradcam"):
        ...

    def generate(self, config, image_path):
        ...
```

Register the engine in `advanced_visualization/models/registry.py`:

```python
from advanced_visualization.models.my_model_type.gradcam import MyModelGradcamEngine

_ENGINES = {
    UniRepLKNetGradcamEngine.name: UniRepLKNetGradcamEngine(),
    MyModelGradcamEngine.name: MyModelGradcamEngine(),
}
```

Then add the model type to `model_type_options` in
`advanced_visualization/settings.json`:

```json
"model_type_options": ["unireplknet", "my_model_type"]
```

## Feature Extraction Compatibility

`advanced_visualization/core/feature_extraction.py` currently assumes this
PyTorch interface:

```python
features = model.feature_extractor(images)
outputs = model.mlp_head(features)
```

For a model that does not expose these attributes, update feature extraction to
delegate feature and prediction logic to the model-specific engine. Until that
adapter exists, such models may generate Grad-CAM but fail during the feature
extraction step.

Temporary workaround:

```bash
python advanced_visualization/cli/prepare_all.py --model-key my_model --skip-features
```

## TensorFlow Support Requirements

TensorFlow is not supported by the current codebase because:

- project training and inference code imports PyTorch modules;
- checkpoints are loaded with `torch.load`;
- datasets and transforms return Torch tensors;
- Grad-CAM hooks use PyTorch forward/backward hooks;
- feature extraction calls PyTorch-specific attributes.

To support TensorFlow models, add a framework-neutral adapter interface first.
The adapter should provide:

- model loading from TensorFlow checkpoint or SavedModel;
- image preprocessing compatible with the trained TensorFlow model;
- prediction output as NumPy or pandas-compatible arrays;
- embedding extraction for feature visualization;
- saliency or Grad-CAM equivalent for visualization overlays;
- registration by `model_type`, similar to the current Grad-CAM registry.

After that, the pipeline can route PyTorch models to the existing path and
TensorFlow models to the TensorFlow adapter.

## Comfortable TensorFlow Workaround: Artifact-Only Integration

TensorFlow can be plugged into the viewer comfortably if TensorFlow work is run
outside this repo and only its artifacts are handed to AutoTorch visualization.
In this mode, AutoTorch does not load the TensorFlow model. It only reads CSVs
and image files.

Recommended artifact layout:

```text
/path/to/tf_model_artifact/
  prepared_predictions.csv
  visualization_manifest.json
  gradcam/
    ...
```

`prepared_predictions.csv` should contain:

- an image path column, for example `absolute_ori_path` or `image_path`;
- a truth column, usually `label`;
- a TensorFlow prediction column with `pred`, `prob`, `score`, or `result` in
  the column name, for example `tf_pred`;
- optional metadata columns for filtering and breakdowns;
- optional feature columns prefixed with `feature_`, `feat_`, `embedding_`, or
  `emb_` for feature-space visualization;
- optional Grad-CAM path column, for example `tf_gradcam_path`.

The easiest Grad-CAM route is to write a direct Grad-CAM path column into the
CSV. The image-review sidebar can select that column as `Grad-CAM path column`,
so the files can be named however the TensorFlow script wants.

If you prefer a Grad-CAM directory instead of a path column, the viewer can also
discover files by image stem:

```text
gradcam/<original_image_stem>.png
gradcam/<original_image_stem>_gradcam.png
gradcam/<original_image_stem>_overlay.png
```

For stronger compatibility with the existing prepared Grad-CAM cache, name files
using the digest convention from
`advanced_visualization/core/gradcam_cache.py`, but this is optional when a
direct path column is present.

Then add a settings entry with the precomputed artifact path:

```json
{
  "key": "my_tensorflow_model",
  "prediction_csv": "/path/to/tf_model_artifact/prepared_predictions.csv",
  "feature_csv": "/path/to/tf_model_artifact/prepared_predictions.csv",
  "artifact_dir": "/path/to/tf_model_artifact",
  "checkpoint": "",
  "weights_epoch": null,
  "model_type": "artifact_only",
  "model_name": "external_tensorflow",
  "head_type": "",
  "image_size": 512,
  "image_column": "absolute_ori_path",
  "prediction_column": "tf_pred",
  "enabled": true
}
```

Do not run `prepare_all.py` for this model unless an `artifact_only` pipeline
mode is added. `prepare_all.py` skips `artifact_only` models because there is no
AutoTorch model to load. Use the Streamlit app directly:

```bash
streamlit run advanced_visualization/unified_app.py
```

This route supports review, filtering, failure buckets, original/Grad-CAM image
comparison, and feature-space exploration from precomputed TensorFlow outputs.
It does not support live TensorFlow inference or live TensorFlow Grad-CAM
generation inside AutoTorch.

## IDRecapture Server Integration

For `/home/jingjie/Dev/automl_platform/idrecapture_server`, keep TensorFlow
model loading, inference, feature extraction, and Grad-CAM generation inside
that repository. AutoTorch should receive only a viewer artifact directory.

Recommended exporter location in the TensorFlow repo:

```text
idfraud/export_autotorch_viz_artifact.py
```

Recommended output:

```text
/mnt3/auto-ekyc/idrecapture/artifacts/parallel/<run>/autotorch_viz/<dataset>/
  prepared_predictions.csv
  visualization_manifest.json
  gradcam/
    <uuid_or_stem>_ori_gradcam.png
    <uuid_or_stem>_crop_gradcam.png
```

The exporter should consume existing IDRecapture outputs:

```text
<model_dir>/infer_results/<dataset>/<dataset>.csv
<model_dir>/infer_results/<dataset>/gradcam/heatmap_<dataset>.csv
<model_dir>/infer_results/<dataset>/gradcam/ori_heatmap_npy/*.npy
<model_dir>/infer_results/<dataset>/gradcam/crop_heatmap_npy/*.npy
```

For the Ench21 parallel Keras model, the important known paths are:

```text
/mnt3/auto-ekyc/idrecapture/artifacts/parallel/Ench21_v1_20251010-1106
/mnt3/auto-ekyc/idrecapture/artifacts/parallel/Ench21_v1_20251010-1106/parallel_Ench21_v1_20251007-0040_Ench21_v1_20251006-1529.h5
```

The prepared CSV should preserve IDRecapture metadata and add AutoTorch-friendly
columns:

- `uuid`
- `label`
- `absolute_ori_path`
- `absolute_ocr_path`
- `tf_parallel_pred`: mapped from IDRecapture `ypred_raw`
- `tf_crop_pred`: mapped from `crop_ypred_raw`
- `tf_ori_pred`: mapped from `ori_ypred_raw`
- `tf_ori_gradcam_path`: PNG overlay path for original image Grad-CAM
- `tf_crop_gradcam_path`: PNG overlay path for crop/OCR image Grad-CAM
- optional `feature_0000`, `feature_0001`, etc. from `avg_pool` or pooled
  `norm4`

Use `model_type: artifact_only` in AutoTorch settings for this output. The
viewer can then select either `tf_ori_gradcam_path` or `tf_crop_gradcam_path` as
the Grad-CAM path column.

This keeps the infrastructure clean:

- IDRecapture Server owns TensorFlow/Keras loading and Ench21-specific path
  resolution.
- AutoTorch visualization owns browsing, filtering, failure buckets, feature
  projection, and image comparison.
- No TensorFlow dependency is needed in the AutoTorch Streamlit runtime.
