#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

LOG_FILE="${LOG_FILE:-gradcam.log}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SAVE_WORKERS="${SAVE_WORKERS:-32}"
CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

export CUDA_DEVICE_ORDER
export CUDA_VISIBLE_DEVICES

CSVS=(
  "feature_visualization/output/Ex8point2res1024_moredata_largerbs_UniRepLKNet_T_legacy_v1_1024_ori_epoch11_full_features.csv"
  "feature_visualization/output/Ex8point2_UniRepLKNet_T_legacy_v1_512_ori_epoch10_full_features.csv"
  "feature_visualization/output/Ex8point4_UniRepLKNet_B_in22k_legacy_v1_512_crop_epoch7_full_features.csv"
)

for csv in "${CSVS[@]}"; do
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] pregenerate Grad-CAM: ${csv}" | tee -a "${LOG_FILE}"
  python advanced_visualization/pregenerate_gradcam.py \
    --csv "${csv}" \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --save-workers "${SAVE_WORKERS}" \
    2>&1 | tee -a "${LOG_FILE}"
done
