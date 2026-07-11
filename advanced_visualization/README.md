# AutoTorch Visualization

Unified Streamlit visualization for ID-fraud model review.

Run:

```bash
streamlit run advanced_visualization/unified_app.py
```

The sidebar `Page` selector contains:

- `Image review`: prepared artifact gallery for predictions, prepared Grad-CAMs, filters, and failure buckets.
- `Feature space`: embedding/projection explorer.
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

Build and run:

```bash
docker compose -f advanced_visualization/docker-compose.yml up --build advanced-visualization
```

Open:

```text
http://localhost:8501
```

The compose file mounts the repo at `/app`. If your configured artifact paths
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
