# AutoTorch Visualization

## Fast Web App

The recommended high-performance viewer is a browser application backed by
FastAPI. It does not use Streamlit:

```bash
python -m advanced_visualization.web.app
```

Open `http://localhost:8000`.

The implementation is intentionally separated by responsibility:

```text
advanced_visualization/web/app.py          HTTP routes only
advanced_visualization/web/models.py       request/response contracts
advanced_visualization/web/repository.py   source discovery and CSV cache
advanced_visualization/web/filtering.py    review filtering and paging
advanced_visualization/web/projections.py  projection computation/cache
advanced_visualization/web/images.py       image validation/thumbnails
advanced_visualization/web/static/         browser interface
```

CSV dataframes are cached by file modification time, feature columns are loaded
as `float32`, filtering and paging happen on the server, images are lazy-loaded,
and generated thumbnails are cached. Feature projections are also cached by
dataset version and projection parameters.

Review and feature data use separate caches. Image review excludes embedding
columns entirely, so opening a prepared CSV with hundreds of feature columns
does not load the feature matrix. The matrix is loaded only when a projection is
requested.

The web viewer supports simultaneous categorical filters with multi-value
checkboxes. Filters carry from a prepared source to its matching feature export.
Feature Space provides PCA, t-SNE, UMAP, and LDA plus fullscreen plotting and a
selected-point inspector for original images, prepared Grad-CAM, Grad-CAM++, and
explicit montage artifacts. The filter sidebar can be collapsed on desktop and
opens as a drawer on mobile.

The feature canvas supports cursor-centered wheel zoom, drag-to-pan, reset,
hover and click inspection, and clickable subclass visibility controls. Its
sidebar reports the live visible-image count and can deterministically cap each
selected color subclass independently before applying the global projection-row
limit. The shared subclass slider provides a quick baseline, with per-subclass
sliders directly below it for targeted overrides.

The image-review grid adapts its effective columns, preview height, and thumbnail
resolution to the number and width of visible cards. Clicking any review or
feature-inspector image opens the shared zoom viewer with wheel, button,
double-click, keyboard, and drag-to-pan controls. The viewer loads a 2K preview
first and requests 4K detail only after zooming in.

Build the standalone web image with:

```bash
docker build -f advanced_visualization/Dockerfile.web -t autotorch-visualization-web .
docker run --rm -p 8000:8000 \
  -v /mnt5:/mnt5 -v /routine_data:/routine_data \
  autotorch-visualization-web
```

## Legacy Streamlit App

Unified Streamlit visualization for ID-fraud model review.

Run:

```bash
streamlit run advanced_visualization/unified_app.py
```

The sidebar `Page` selector contains:

- `Image review`: prepared artifact gallery for predictions, prepared Grad-CAMs, filters, and failure buckets.
- `Feature space`: embedding/projection explorer.
- `Launch workspace`: settings-driven model-specific viewer clones such as VAN Small or UniRepLKNet workspaces.
- `Settings`: prediction CSV and model artifact configuration.

## Settings

Use the `Settings` page as the source of truth. It writes:

```text
advanced_visualization/settings.json
```

Define:

- prediction CSV
- model key
- artifact directory
- model type, currently `unireplknet` or `artifact_only`
- prediction column

The Settings page separates viewer data sources from CLI-only model-loading
fields. The Streamlit viewer does not load checkpoint weights. Checkpoint path,
weights epoch, model name, head type, and image size are only used by
`prepare_all.py` / `pregenerate_gradcam.py`.

The app no longer hardcodes model runs or a default prediction CSV. Configure
them in the `Settings` page or edit `advanced_visualization/settings.json`.

## Unified Pipeline

The preferred preparation path is the single pipeline:

```bash
python advanced_visualization/cli/prepare_all.py
```

It uses enabled models from `advanced_visualization/settings.json` and performs:

1. standard artifact setup
2. prepared CSV creation
3. model forward pass for feature columns and prediction column
4. prepared Grad-CAM overlay generation
5. manifest writing

You can also run one configured model:

```bash
python advanced_visualization/cli/prepare_all.py --model-key my_model_key
```

Useful options:

```bash
--skip-features
--skip-gradcam
--limit 200
--batch-size 8
--gradcam-batch-size 16
```

The `Settings` page only edits configuration. Run preparation from the CLI so the viewer never generates Grad-CAM during browsing.

## Feature Space

Feature-space exploration is now part of this package:

```text
advanced_visualization/views/feature_space.py
```

Legacy helper scripts were moved to:

```text
advanced_visualization/cli/extract_unireplknet_t_features.py
advanced_visualization/cli/create_feature_csv_template.py
```

Existing feature CSVs are stored under:

```text
advanced_visualization/output/
```

## Launchable Workspaces

Model-specific viewer clones are launched from `Launch workspace`.
Supported model types are configured in `advanced_visualization/settings.json`
under `extra_view_configs`.

Each config defines:

- model type
- branches, such as crop/ori/image
- available Grad-CAM layers or montage outputs
- Grad-CAM/montage path column template or explicit column candidates
- default layer and prediction columns

This is artifact-first. New model families normally do not need new Streamlit
code if they export a prepared CSV with image paths, predictions, optional
`feature_0000...` columns, and PNG explanation path columns. A model can export
single-layer Grad-CAM PNGs or pre-rendered montage PNGs. Add or edit an
`extra_view_configs` entry when the model needs different branches, layer
options, or montage columns.

The current VAN Small TensorFlow handoff is montage-only. Its launch workspace
expects:

```text
tf_crop_layer_montage_path
tf_ori_layer_montage_path
```

Per-layer VAN Small Grad-CAM path columns are optional for this delivery.

Each launched workspace provides its own model-specific clone of:

- image review
- feature-space projection
- configured layered Grad-CAM inspection

The launcher opens workspaces through a query-param URL in a new tab, so the
launched workspace is independent from the current app page and does not change
the main viewer's selected page.

## Lower-Level Commands

Manifest and prepared CSV only:

```bash
python advanced_visualization/cli/preparation.py \
  --artifact-dir /path/to/model_artifact_dir \
  --pred-csv /path/to/predictions.csv \
  --weights-epoch 11 \
  --model-key my_model_key
```

Grad-CAM preparation only:

```bash
python advanced_visualization/cli/pregenerate_gradcam.py \
  --csv /path/to/model_artifact_dir/prepared_predictions.csv \
  --limit 20
```

## Docker

Build and run the recommended FastAPI web viewer:

```bash
docker compose -f advanced_visualization/docker-compose.yml up --build advanced-visualization
```

Open:

```text
http://localhost:8000
```

Compose builds `Dockerfile.web` and mounts the repo at `/app`, so the container
runs the current web implementation. If your configured artifact paths
live outside the repo, add the required bind mount in
`advanced_visualization/docker-compose.yml`.

## Model-Specific Code

Model-specific Grad-CAM implementations live under:

```text
advanced_visualization/models/<model_type>/gradcam.py
```

Current folder:

- `models/unireplknet/gradcam.py`

Future model families should add their own folder and register the engine in
`advanced_visualization/models/registry.py`. Model-specific behavior includes
model construction, checkpoint loading, preprocessing, target layer selection,
score selection, CAM computation, and any model-specific feature extraction
logic.

For the full checklist for adding a new run, new PyTorch backbone, new
visualization model family, or TensorFlow support, see
`advanced_visualization/handoff_docs/01_autotorch_model_integration_guide.md`.
