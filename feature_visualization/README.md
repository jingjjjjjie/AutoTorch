# UniRepLKNet-T Feature Visualization

Interactive item-level feature-space explorer for UniRepLKNet-T embeddings.

The workflow is intentionally CSV-first:

1. Export features from images and a trained AutoTorch checkpoint.
2. Launch the Streamlit app with the item-level feature CSV.
3. Select feature columns, class columns, merge mappings, and subsets interactively.

## Expected CSV

Each row is one item/sample/image. Minimum columns:

- One or more numeric feature columns. By convention the extractor writes `feature_0000` ... `feature_0639`.
- At least one categorical column for grouping, for example `label`, `source`, `batch`, `sample_type`, or `class`.

Recommended columns:

- `path`: image path for previewing selected points.
- `id`: stable row identifier.
- `split`, `batch`, `source`, `sample_type`, `class`: metadata used for filtering and color/group comparisons.

Create a template if you want the expected shape first:

```bash
python feature_visualization/create_item_csv_template.py \
  --output /tmp/unireplknet_t_item_template.csv
```

## Export UniRepLKNet-T features

```bash
python feature_visualization/extract_unireplknet_t_features.py \
  --csv /path/to/items.csv \
  --checkpoint /path/to/run/Ex8point2_UniRepLKNet_T_legacy_v1_1024_ori.pt \
  --output /path/to/unireplknet_t_features.csv \
  --image-column path \
  --label-column label \
  --head-type legacy_v2 \
  --image-size 1024 \
  --batch-size 8
```

The extractor treats each input row as one item, preserves all input CSV columns, and appends UniRepLKNet-T feature columns.

For the joined annotation CSV used in this branch:

```bash
python feature_visualization/extract_unireplknet_t_features.py \
  --csv /home/jingjie/AutoTorch/src/eval/idfraud/annotation/joined_predictions.csv \
  --checkpoint /mnt3/repo_and_weights/runs2/Ex8point2res1024_moredata_largerbs_UniRepLKNet_T_legacy_v1_1024_ori/checkpoints/epoch_11.pt \
  --output /home/jingjie/AutoTorch/feature_visualization/output/joined_predictions_unireplknet_t_epoch11_features.csv \
  --image-column absolute_ori_path \
  --label-column label \
  --head-type legacy_v1 \
  --output-type probs \
  --prediction-column Ex8point2res1024largerbs_pred_ori_rerun \
  --image-size 1024 \
  --batch-size 8 \
  --num-workers 8 \
  --device cuda:2 \
  --limit-class-column Recapture_Subclass \
  --limit-class-value Genuine \
  --limit-class-rows 5000
```

## Launch the app

```bash
streamlit run feature_visualization/app.py
```

To preload the generated feature CSV:

```bash
AUTOTORCH_FEATURE_CSV=/home/jingjie/AutoTorch/feature_visualization/output/joined_predictions_unireplknet_t_epoch11_features.csv \
streamlit run feature_visualization/app.py
```

Upload a feature CSV in the sidebar. The app supports:

- PCA or t-SNE projections.
- Toggle/filter any categorical class or batch column.
- Merge classes using explicit rules such as `colour printed=collected colour printed,printed colour prod`.
- Color, symbol, and facet controls for different metadata views.
- Minimal point hover with image preview from the item inspector when a valid image path column is available.

## Merge Mapping Format

Use one rule per line:

```text
colour printed=collected colour printed,printed colour prod
mix and match=mix,match
```

Values not listed in a rule keep their original value, so merges are reversible and auditable.
