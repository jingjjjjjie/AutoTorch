# Adding Models to Advanced Visualization

This guide explains how new models plug into AutoTorch advanced visualization.

The long-term design is artifact-first. The viewer can support almost any model
family if the model owner exports a compatible artifact:

- a prepared CSV with image paths, labels, predictions, and metadata;
- optional numeric feature columns named `feature_0000`, `feature_0001`, etc.;
- Grad-CAM or explanation image path columns;
- an `extra_view_configs` entry when the model needs model-specific layer or
  branch controls.

Model-specific Python code is only required when AutoTorch itself must load the
model and generate predictions, features, or Grad-CAM. If another infra already
exports the artifact, use `model_type: artifact_only` and configure the extra
view from settings.

## Current Support

The Streamlit viewer is artifact-first: it reads prepared CSVs, feature columns,
manifests, and Grad-CAM image files. The model framework can be PyTorch,
TensorFlow, Keras, ONNX, or something else, as long as the exported artifact
matches the viewer schema.

The AutoTorch preparation pipeline supports PyTorch models that can be loaded by
AutoTorch. This is the path used when AutoTorch performs feature extraction and
Grad-CAM generation itself.

AutoTorch does not need TensorFlow/Keras runtime support for artifact-only
TensorFlow models. TensorFlow support is only needed if AutoTorch is expected to
load TensorFlow models directly.

## Universal Artifact Contract

Any model can be visualized if it exports a CSV with the following practical
shape:

```text
uuid or item_id
label
one or more image path columns
one numeric prediction column
optional metadata columns
optional feature_0000...feature_N columns
optional Grad-CAM/explanation PNG path columns
```

Recommended column conventions:

- Image columns: `absolute_ori_path`, `absolute_ocr_path`, `image_path`, `path`.
- Prediction columns: include `pred`, `prob`, `score`, or `result` in the name.
- Feature columns: `feature_0000`, `feature_0001`, ...
- Explanation columns: direct PNG paths, for example
  `tf_crop_norm3_logit_gradcam_path`.

If the CSV follows this contract:

- `Image review` can browse images, predictions, failures, filters, and prepared
  Grad-CAMs.
- `Feature space` can project any numeric `feature_...` columns.
- `Launch workspace` can render model-specific viewer clones configured in
  `extra_view_configs`.

## What Changes Per Model

For artifact-only models, the model differences are mostly configuration:

- branches, such as `image`, `crop`, `ori`, `front`, `back`;
- Grad-CAM/explanation layers;
- Grad-CAM path column template;
- default layer;
- prediction and metadata column candidates.

These are configured in:

```text
advanced_visualization/settings.json
extra_view_configs
```

In most cases, adding a new artifact-only model should not require new Python
code.

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

Create a new model-specific engine only when AutoTorch needs to load the model
and generate artifacts itself.

If the model's own infra exports a prepared CSV, features, and explanation PNG
paths, skip this section and use `artifact_only` plus `extra_view_configs`.

When AutoTorch must generate artifacts, create a new model-specific engine when
the new model needs different loading, target-layer selection, score logic, CAM
computation, or feature extraction.

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

Direct TensorFlow model loading is not supported by the current AutoTorch
pipeline because:

- project training and inference code imports PyTorch modules;
- checkpoints are loaded with `torch.load`;
- datasets and transforms return Torch tensors;
- Grad-CAM hooks use PyTorch forward/backward hooks;
- feature extraction calls PyTorch-specific attributes.

To support TensorFlow models directly inside AutoTorch, add a framework-neutral
adapter interface first. The adapter should provide:

- model loading from TensorFlow checkpoint or SavedModel;
- image preprocessing compatible with the trained TensorFlow model;
- prediction output as NumPy or pandas-compatible arrays;
- embedding extraction for feature visualization;
- saliency or Grad-CAM equivalent for visualization overlays;
- registration by `model_type`, similar to the current Grad-CAM registry.

After that, the pipeline can route PyTorch models to the existing path and
TensorFlow models to the TensorFlow adapter.

For normal visualization usage, prefer artifact-only TensorFlow integration:
run TensorFlow inference and Grad-CAM in the TensorFlow infra, export the CSV and
PNG paths, and let AutoTorch read the artifact.

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

## Launchable Workspaces

Model-specific inspection tools should be added through the workspace launcher,
not by adding model-specific controls to the general image review.

Current launcher entry:

```text
advanced_visualization/views/launcher.py
```

Current workspace composer:

```text
advanced_visualization/views/extra_views/workspace.py
```

Current extra-view registry:

```text
advanced_visualization/views/extra_views/registry.py
```

Current generic layered Grad-CAM implementation used inside each workspace:

```text
advanced_visualization/views/extra_views/layered_gradcam.py
```

Extra views are configured in `advanced_visualization/settings.json` under:

```text
extra_view_configs
```

The default supported model types are:

```text
vansmall
unireplknet
```

Each config defines branches, layers, and a Grad-CAM path column template. For
example, the VAN Small config uses:

```text
tf_{branch}_{layer}_{score}_gradcam_path
```

Generic config shape:

```json
{
  "model_type": "my_model",
  "label": "My Model Grad-CAM Review",
  "description": "Layered explanation review for My Model artifacts.",
  "view": "layered_gradcam",
  "score": "logit",
  "column_template": "my_{branch}_{layer}_{score}_gradcam_path",
  "required_columns": [
    "my_image_stage4_logit_gradcam_path"
  ],
  "branches": [
    {
      "key": "image",
      "label": "Image",
      "image_candidates": ["image_path", "absolute_ori_path", "path"]
    }
  ],
  "layers": [
    {"key": "stage3", "label": "stage3"},
    {"key": "stage4", "label": "stage4"}
  ],
  "default_layer": "stage4",
  "prediction_candidates": ["my_pred", "pred", "score"],
  "metadata_columns": ["Recapture_Subclass", "Quality_Issue"]
}
```

The layered view dynamically builds its layer controls from this config. A model
with one layer and a model with ten layers use the same view code.

Each launched workspace provides:

- its own image review over the selected CSV/artifact;
- its own feature-space projection over the same CSV/artifact;
- its own configured layered Grad-CAM inspection.

The launcher generates a query-param URL and opens the workspace in a new tab.
This keeps the launched model-specific workspace independent from the main app
page/session state.

The VAN Small workspace expects the TensorFlow exporter to provide at least:

```text
tf_crop_norm3_logit_gradcam_path
tf_ori_norm3_logit_gradcam_path
```

Optional VAN Small layer comparison columns:

```text
tf_crop_block3_3_logit_gradcam_path
tf_ori_block3_3_logit_gradcam_path
tf_crop_block4_1_logit_gradcam_path
tf_ori_block4_1_logit_gradcam_path
tf_crop_norm4_logit_gradcam_path
tf_ori_norm4_logit_gradcam_path
tf_crop_layer_montage_path
tf_ori_layer_montage_path
```

Use:

```bash
streamlit run advanced_visualization/unified_app.py
```

Then open `Launch workspace`, select the prepared CSV/artifact source, and
choose the configured model type and workspace.
