# IDRecapture Repository Understanding

Last reviewed: 2026-07-10

## User Intent

Understand this repository without changing application code. Preserve context for future work to build Grad-CAM and visualization tooling similar to `/home/jingjie/AutoTorch/advanced_visualization`.

Primary model to focus on:

```text
/mnt3/auto-ekyc/idrecapture/artifacts/parallel/Ench21_v1_20251010-1106/parallel_Ench21_v1_20251007-0040_Ench21_v1_20251006-1529.h5
```

This is a parallel merged Keras `.h5` model built from:

- crop model: `Ench21_v1_20251007-0040`, best checkpoint `/mnt3/auto-ekyc/idrecapture/artifacts/crop/Ench21_v1_20251007-0040/checkpoints/checkpoint_15.h5`
- ori model: `Ench21_v1_20251006-1529`, best checkpoint `/mnt3/auto-ekyc/idrecapture/artifacts/ori/Ench21_v1_20251006-1529/checkpoints/checkpoint_18.h5`
- merge type: `average`
- merged artifact directory: `/mnt3/auto-ekyc/idrecapture/artifacts/parallel/Ench21_v1_20251010-1106`
- merge metadata: `/mnt3/auto-ekyc/idrecapture/artifacts/parallel/Ench21_v1_20251010-1106/logs/merge/info.json`

Both submodels were trained with:

- `model_name`: `vansmall`
- input size: `[512, 512]`
- batch size: `16`
- epochs configured: `100`
- model checkpoint: `null`, trained from scratch
- train/validation split: `0.9`
- TensorFlow version recorded in logs: `2.8.0`

## Repository Shape

This repo is a FastAPI-backed TensorFlow/Keras training, evaluation, merging, leaderboard, and Grad-CAM service for ID recapture detection.

Important entrypoints:

- `server.py`: FastAPI app wiring routes.
- `route/train_api.py`: training endpoints.
- `route/eval_api.py`: evaluation, parallel merge, parallel evaluation, leaderboard, and Grad-CAM endpoints.
- `assets/id_assets.py`: launches train/eval/merge/Grad-CAM scripts as subprocesses and handles data processing helpers.
- `config/idrecapture_config.yml`: main root paths, data batch records, train/eval/parallel/leaderboard/Grad-CAM default config.
- `config/config.py`: YAML config loader/writer.

Core scripts:

- `idfraud/main_train.py`: training orchestration, GPU selection, data loading, strategy setup, calls `ModelTrain`.
- `idfraud/main_eval.py`: single ori/crop evaluation.
- `idfraud/main_eval_parallel.py`: merged model evaluation.
- `idfraud/main_parallel.py`: two-model merge entrypoint.
- `idfraud/main_mix_parallel.py`: three-model/mixed merge entrypoint.
- `idfraud/main_gradcam.py`: existing batch Grad-CAM generation for parallel models.

Core model modules:

- `idfraud/model/load_model.py`: builds/loads single models. `vansmall` comes from `idfraud.tfvan.VanSmall`.
- `idfraud/model/train.py`: training callbacks and model fit flow.
- `idfraud/model/evaluate.py`: evaluation preprocessing, inference, output CSVs, threshold metrics.
- `idfraud/model/merge_model.py`: merges crop and ori models into a two-input model.
- `idfraud/model/data_loader.py`: CSV-backed TensorFlow dataset loader.
- `idfraud/model/model_preprocess.py`: shared image reshape/scale/color preprocessing.
- `idfraud/gradcam.py`: existing simple TensorFlow Grad-CAM implementation.

Custom VAN implementation:

- `idfraud/tfvan/model.py`: VAN model builder.
- `idfraud/tfvan/block.py`: VAN blocks.
- `idfraud/tfvan/attn.py`: spatial attention layers.
- `idfraud/tfvan/embed.py`, `norm.py`, `mlp.py`, `drop.py`: supporting layers.

## Data And Path Conventions

Configured roots in `config/idrecapture_config.yml`:

- `mnt_path`: `/mnt/auto-ekyc/idrecapture`
- `mnt2_path`: `/mnt2/auto-ekyc/idrecapture`
- `artifact_path`: `/mnt3/auto-ekyc/idrecapture/artifacts`
- `primary_dataset_path`: `/mnt3/auto-ekyc/idrecapture/datasets`
- `secondary_dataset_path`: `/mnt2/raw_dataset/research/idv/idfraud/data/`
- `live_path`: `/mnt/auto-ekyc/live_data/`

`Utils.read_data(image_type, batch_list, data_type, ...)` combines CSVs and creates a unified `path` column:

- crop uses `ocr_path`
- ori uses `ori_path`
- corner uses `corner_path`
- labels are binary: `genuine -> 0`, anything else -> `1`
- training data is split into `dataset_type` train/validation
- paths are validated across configured roots

Evaluation path lookup in `EvaluateModel` checks:

1. `root_path.mnt_path`
2. `root_path.primary_dataset_path`
3. `root_path.secondary_dataset_path/image_source`
4. `root_path.live_path`

## Preprocessing

The relevant preprocessing path for Ench21 is in `ModelPreprocess._reshape_image`:

- Input is RGB.
- For non-distorted resize with `square=True`, if image height is greater than or equal to width, it center-crops to a square first.
- It then uses `tf.image.resize_with_pad(height, width)` unless `mirror_pad=True`.
- Ench21 uses `[512, 512]`.
- No explicit ImageNet normalization is applied in the repo path inspected; images remain in pixel-like TensorFlow float values after resize.

Training loader defaults in `CSVTFDatasetLoader`:

- `distortion=False`
- `convert_colour_space='rgb'`
- `square=True`
- `mirror_pad=False`
- `grayscale=False`
- `cie_color=False`
- `one_hot_encoding=False`

Evaluation uses `_fetch_and_preprocess_tf`, which follows the same 512 square/pad behavior.

## Ench21 Parallel Model Architecture

The target H5 file was inspected with `h5py` metadata, not fully loaded into TensorFlow.

Observed structure:

- root H5 key: `model_weights`
- model has two inputs:
  - `input_1`: `[None, 512, 512, 3]`
  - `input_2`: `[None, 512, 512, 3]`
- nested submodels:
  - `cropped_model`
  - `ori_model`
- merge layer:
  - `average`
- output also includes:
  - `concatenate`, used by this repo to expose raw crop and ori scores beside the merged score

The merge code in `idfraud/model/merge_model.py` constructs the model as:

```python
inputs = [Input(shape=crop_input_shape), Input(shape=ori_input_shape)]
x_crop = crop_model(inputs[0])
x_ori = ori_model(inputs[1])
x = Average()([x_crop, x_ori])
combined_output = Concatenate()([x_crop, x_ori])
model = Model(inputs=inputs, outputs=[x, combined_output])
```

The evaluation code feeds parallel inputs in this order:

```python
inputs = [crop_batch, ori_batch]
```

For result CSVs, parallel evaluation writes:

- `ypred_raw`: merged output
- `ypred`: thresholded merged output
- `crop_ypred_raw`: first raw score from concatenated output
- `ori_ypred_raw`: second raw score from concatenated output

## VAN Small Layer Notes For Grad-CAM

`ModelLoader._vansmall` builds:

1. `VanSmall(include_top=False, classifier_activation=None)`
2. interpolate positional embeddings for 512 or 768 input
3. call VAN
4. add `GlobalAveragePooling2D(name='avg_pool')`
5. add dropout/dense classifier head
6. final `Dense(..., activation='sigmoid', name='pred')`

In `idfraud/tfvan/model.py`, `Van(..., include_top=False)` returns the `norm4` output by default. For `vansmall`, expected stage depths are `(2, 2, 4, 2)` and final spatial features pass through:

- `patch_embed4`
- `block4.0`
- `block4.1`
- `norm4`
- repo-added `avg_pool`
- dense/dropout head
- `pred`

The current default Grad-CAM config in `config/idrecapture_config.yml` uses `last_conv_layer: conv5_block3_out`, which is ResNet-specific and not correct for this Ench21 VAN Small model. For future Ench21 Grad-CAM, likely target candidates are:

- submodel layer `norm4` for last spatial VAN feature map
- possibly `block4.1` if targeting before final normalization
- possibly nested attention/conv layers if a lower-level heatmap is wanted

Need verify exact Keras layer access syntax after loading the H5, because `norm4` exists inside both `cropped_model` and `ori_model`. A future implementation should load the parallel model, then select:

```python
parallel = tf.keras.models.load_model(path, custom_objects=...)
crop_model = parallel.get_layer("cropped_model")
ori_model = parallel.get_layer("ori_model")
target = crop_model.get_layer("norm4")  # or ori_model.get_layer("norm4")
```

Use custom objects when fully loading:

```python
{
    "PatchEmbedding": PatchEmbedding,
    "Block": Block,
    "LayerNorm": LayerNorm,
}
```

If loading the merged H5 fails on custom objects, also include `WeightedAverage` only for weighted-average models. Ench21 uses standard `Average`.

## Existing Grad-CAM Flow

`idfraud/gradcam.py` provides:

- `GradCAM.set_gradcam_model(last_conv_layer_name)`
- `extract_gradcam_heatmap`
- `jet_heatmap`
- `visualize_gradcam`
- `generate_gradcam`

Current limitations:

- assumes a single submodel, not the two-input merged model directly
- `set_gradcam_model` uses `self.model.get_layer(last_conv_layer_name)` and `self.model.output`
- `extract_gradcam_heatmap` assumes a simple output tensor and `preds[:, pred_index]`
- normalizes heatmap by max without guarding zero max
- returns `pred_label = preds.numpy()[0][0]`
- existing `main_gradcam.py` extracts `cropped_model` or `ori_model` from the parallel H5 and generates `.npy` heatmaps, so it avoids explaining the merged output directly

`idfraud/main_gradcam.py` expects an evaluated parallel result CSV under:

```text
{model_dir}/infer_results/{dataset}/{dataset}.csv
```

It writes:

```text
{model_dir}/infer_results/{dataset}/gradcam/ori_heatmap_npy/*.npy
{model_dir}/infer_results/{dataset}/gradcam/crop_heatmap_npy/*.npy
{model_dir}/infer_results/{dataset}/gradcam/heatmap_{dataset}.csv
```

For the Ench21 VAN model, the user will likely want an improved path:

- target `norm4`, not `conv5_block3_out`
- support both ori and crop submodels
- optionally generate overlay PNGs, not only `.npy`
- support interactive item inspection similar to AutoTorch Streamlit app
- consider Grad-CAM++ support like the reference app

## AutoTorch Feature Visualization Reference

Reference directory:

```text
/home/jingjie/AutoTorch/advanced_visualization
```

Important files:

- `README.md`: CSV-first feature exploration workflow.
- `cli/extract_unireplknet_t_features.py`: exports item-level features and predictions from a PyTorch model.
- `app.py`: Streamlit feature-space explorer with image proxy and Grad-CAM/Grad-CAM++ generation.
- `cli/create_feature_csv_template.py`: creates expected CSV shape.

Reference app capabilities to mirror later:

- CSV-first workflow with metadata preserved.
- Auto-detect numeric feature columns by prefixes like `feature_`, `feat_`, `embedding_`, `emb_`.
- PCA, t-SNE, UMAP, LDA projections.
- Metadata filtering, class merge rules, color/symbol/facet controls.
- Image preview via local HTTP image proxy.
- Grad-CAM and Grad-CAM++ buttons for selected point.
- Cached model bundles and cached Grad-CAM overlay output.
- Secure-ish allowed image roots for proxy.

Reference Grad-CAM design:

- model config keyed by active feature CSV stem
- lazy-load model bundle and cache it
- register forward and backward hooks on target layer
- compute either Grad-CAM or Grad-CAM++
- normalize CAM
- overlay heatmap on original image
- serve output PNG through `/gradcam` endpoint

For TensorFlow/Keras Ench21, equivalent design should use `tf.GradientTape` instead of PyTorch hooks.

## Suggested Future Implementation Direction

For future Grad-CAM and visualization work, avoid modifying training/evaluation behavior first. Build a separate visualization/export path that can consume existing Ench21 result CSVs and the target H5 model.

Recommended pieces:

1. Feature/embedding exporter for Keras Ench21:
   - load target parallel H5
   - extract `cropped_model` and `ori_model`
   - create feature models ending at `avg_pool` or `norm4` + pooling
   - read an eval/result CSV
   - resolve image paths using the same root fallback logic as `EvaluateModel`
   - apply the same 512 preprocessing
   - write CSV with metadata, predictions, and `feature_0000...` columns

2. Grad-CAM generator:
   - load target submodel
   - target `norm4` by default
   - class score is the sigmoid fraud score from `pred`
   - allow `model_view`: `crop`, `ori`, and maybe `parallel`
   - for parallel view, compute separate crop and ori CAMs, and display both because merged average is composed from both branches

3. Streamlit viewer:
   - can be adapted from `/home/jingjie/AutoTorch/advanced_visualization/views/feature_space.py`
   - replace PyTorch model config/load/Grad-CAM functions with TensorFlow/Keras equivalents
   - keep PCA/t-SNE/UMAP/LDA and metadata filtering
   - add image path resolver for repo result CSV columns: `ori_path`, `ocr_path`, `path`, `batch_name`

## Watchouts

- The repo has path inconsistencies: `config.root_path.artifact_path` points to `/mnt3/.../artifacts`, but some existing Grad-CAM code uses `root_path.mnt2_path/artifacts/parallel`. For Ench21, the real artifact is under `/mnt3/.../artifacts/parallel`.
- `server.py` calls `configManager.load_server_config()` in `/version`, but that method was not present in `config/config.py` during review.
- `idfraud/main_gradcam.py` has a probable Pandas bug: `result_df_sub = result_df_sub.loc[result_df_sub['path'].isin(final_result_df_path)]` passes a DataFrame to `isin`; likely intended to use a path Series.
- `main_gradcam.py` prints `alt_row_path` even if the original path exists, which may be undefined.
- `main_gradcam.py` mutates `final_result_df_path['path'][index]`, which may trigger chained-assignment issues.
- Many modules use hard-coded `/app` or `/automl` sys.path assumptions. Local scripts may need `PYTHONPATH` adjustment.
- The H5 was only metadata-inspected in this understanding pass. Full TensorFlow loading should be verified before implementation.

## No Application Code Changed

This note is the only intended repo change from the understanding pass.
