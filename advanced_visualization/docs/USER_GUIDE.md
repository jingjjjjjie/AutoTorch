# User Guide

## Start the primary viewer

From the repository root:

```bash
python -m advanced_visualization.web.app
```

Open <http://localhost:8000>. A successful service also reports `{"status":"ok"}` at <http://localhost:8000/api/health>.

The source selector lists enabled CSVs from `advanced_visualization/settings.json`. If no source appears, verify that the configured CSV exists and the model is enabled.

## Image review

Use this page to inspect one experiment:

1. Select the source and image/prediction/truth columns.
2. Set the decision threshold and truth-row policy.
3. Filter by failure type, categories, or literal text.
4. Choose original, Grad-CAM, or both.
5. Click an image for the shared zoom/pan viewer.

Prepared CAM files are used when available. Browsing does not generate CAMs on demand.

## Feature space

Feature columns must follow a supported numbered pattern such as `feature_0000`, `feat_1`, `embedding_12`, or `emb_12`.

Choose PCA, t-SNE, UMAP, or LDA, then set the color/group and row limits. Filtering happens before deterministic sampling. Click a point to inspect its source image and explanation artifact.

Use PCA for a fast first look. t-SNE is capped at 5,000 rows; UMAP is capped at 50,000 rows and requires `umap-learn`. LDA requires a valid group column with at least two classes.

## Compare models

This page answers both directions of the experiment question:

- what Experiment A gets correct that B gets wrong;
- what Experiment B gets correct that A gets wrong;
- what both get correct or wrong;
- which rows cannot be fairly compared because truth differs, scoring is missing, or the item exists in only one source.

Select A and B, their stable ID/truth/prediction/image columns, and a separate threshold for each model. Matrix cells filter the gallery. Repeated stable IDs are matched by occurrence instead of producing a Cartesian join.

`Open A viewer` and `Open B viewer` create independent review tabs. You can change filters in either viewer without changing the other one.

## Results analysis

This page supports the workflow currently represented by:

```text
src/eval/idfraud/annotation/visualize_and_analyze_results.ipynb
```

For the joined-prediction source, select original and crop predictions/images. The page applies the configurable cleanup rules, computes original/crop/arithmetic-mean metrics, shows subclass/identity/quality breakdowns, and lets you drill into the corresponding image pairs.

The notebook-compatible default decision uses the strict rule `score > threshold`. Confirm the chosen columns and threshold before comparing reported numbers with an older notebook run.

## Configure sources

Each enabled model entry can provide:

- `prediction_csv`: review/analysis predictions;
- `feature_csv`: high-dimensional feature export;
- `artifact_dir`: prepared CSV, manifest, and `gradcam/` directory;
- `image_column` and `prediction_column`: preferred defaults;
- `review_preset`: optional default review controls;
- checkpoint/model fields: only needed by model-loading preparation commands.

The Streamlit Settings page can edit the JSON:

```bash
streamlit run advanced_visualization/unified_app.py
```

Path compatibility includes the known `/routine_data/...` to `/mnt5/routine_data/...` mount alias. Prefer storing paths that are valid in the deployment environment and mount external directories into Docker explicitly.

## Prepare data and Grad-CAM

Complete pipeline:

```bash
python -m advanced_visualization.cli.prepare_all --model-key MODEL_KEY
```

Useful switches include `--skip-features`, `--skip-gradcam`, `--limit`, `--batch-size`, and `--gradcam-batch-size`.

Grad-CAM only:

```bash
python -m advanced_visualization.cli.pregenerate_gradcam \
  --csv /path/to/prepared_predictions.csv \
  --cam-method gradcam \
  --cam-target fraud \
  --only-missing
```

Repeat `--cam-method` or `--cam-target` to generate multiple variants. Use `--dry-run` to count work without writing overlays.

## Troubleshooting

- **No sources:** check `enabled`, CSV paths, and container bind mounts.
- **Images unavailable:** inspect an actual CSV path from the same environment as the server.
- **No feature columns:** verify numbered feature names and numeric values.
- **No prepared CAM:** run preparation, verify `artifact_dir/gradcam`, and match the chosen method/target.
- **Comparison has many missing rows:** confirm both stable ID columns identify the same samples.
- **Metrics differ from the notebook:** confirm cleanup filters, selected truth/prediction columns, and strict threshold behavior.
