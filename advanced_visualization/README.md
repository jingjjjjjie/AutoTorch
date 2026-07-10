# Advanced Visualization

Paged Streamlit image viewer for fast ID-fraud failure review.

It is CSV-first and works with annotation CSVs or feature-export CSVs. The app focuses on:

- Advanced subclass and metadata filtering.
- High-confidence and low-confidence failure buckets.
- False-positive and false-negative review.
- Paged original / Grad-CAM / side-by-side image inspection.
- Subclass breakdown tables for spotting repeated patterns.

## Launch

```bash
streamlit run advanced_visualization/app.py
```

By default, the app lists CSVs from:

```text
/home/jingjie/AutoTorch/feature_visualization/output
```

To preload a CSV:

```bash
AUTOTORCH_ADVANCED_VIS_CSV=/path/to/joined_predictions.csv \
streamlit run advanced_visualization/app.py
```

You can also point `AUTOTORCH_ADVANCED_VIS_CSV` at a directory containing CSV files.

## Expected CSV

Useful columns:

- An image path column, for example `absolute_ori_path`, `absolute_ocr_path`, `path`, or `image_path`.
- A stable ID column, for example `uuid` or `id`.
- A subclass/group column, for example `Recapture_Subclass`, `Tamper_Subclass`, or `Data_Identity`.
- A truth column, usually `label`.
- One numeric prediction column containing probabilities or scores.
- Optional Grad-CAM path columns, or a directory containing Grad-CAM images named by the original image stem.

## Grad-CAM Images

The viewer does not require Grad-CAM paths. If you have precomputed Grad-CAM files, use either:

- `Grad-CAM path column`: a CSV column that points directly to each overlay image.
- `Grad-CAM directory`: a folder where files are named like `<original_image_stem>.png`, `<stem>.jpg`, `<stem>_gradcam.png`, or `<stem>_overlay.png`.

For configured experiment CSVs, missing Grad-CAMs can also be generated on demand in the app, or pre-generated from the CLI. New generated overlays are written to the run artifact directory:

```text
/mnt3/repo_and_weights/runs.../<experiment>/gradcam
```

Pre-generate a small test batch:

```bash
python advanced_visualization/pregenerate_gradcam.py \
  --csv /home/jingjie/AutoTorch/feature_visualization/output/Ex8point2_UniRepLKNet_T_legacy_v1_512_ori_epoch10_full_features.csv \
  --limit 20
```

Pre-generate only a filtered subset:

```bash
python advanced_visualization/pregenerate_gradcam.py \
  --csv /home/jingjie/AutoTorch/feature_visualization/output/Ex8point2res1024_moredata_largerbs_UniRepLKNet_T_legacy_v1_1024_ori_epoch11_full_features.csv \
  --filter Data_Identity=Feb_2026_mixed_RoutineAnnotation \
  --filter Recapture_Subclass=Genuine \
  --limit 200
```

Use `--dry-run` to count what would be generated without writing files. By default, `--only-missing` is enabled.
