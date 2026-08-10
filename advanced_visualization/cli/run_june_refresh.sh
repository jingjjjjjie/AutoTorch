#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="/home/jingjie/AutoTorch"
SOURCE="$REPO_ROOT/src/eval/idfraud/annotation/joined_predictions_30june.csv"
ARTIFACT_ROOT="/mnt4/advanced_visualization"
COMPOSE_FILE="$REPO_ROOT/advanced_visualization/docker-compose.yml"

torch_features() {
  local gpu="$1"
  local model_id="$2"
  local image_column="$3"
  local output="$4"
  local batch_size="$5"
  echo "FEATURE START gpu=$gpu model=$model_id $(date -Is)"
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$gpu" \
    AUTOTORCH_GRADCAM_DEVICE=cuda:0 \
    python -m advanced_visualization.cli.extract_registered_features \
      --model-id "$model_id" \
      --csv "$SOURCE" \
      --output "$output" \
      --incremental-from "$output" \
      --image-column "$image_column" \
      --batch-size "$batch_size" \
      --num-workers 8
  echo "FEATURE DONE gpu=$gpu model=$model_id $(date -Is)"
}

tensorflow_features() {
  local gpu="$1"
  local model_id="$2"
  local image_column="$3"
  local prediction_column="$4"
  local output="$5"
  local gpu_uuid
  gpu_uuid=$(nvidia-smi -i "$gpu" --query-gpu=uuid --format=csv,noheader)
  echo "FEATURE START gpu=$gpu model=$model_id $(date -Is)"
  docker compose -f "$COMPOSE_FILE" run --rm --no-deps \
    -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
    -e CUDA_VISIBLE_DEVICES=0 \
    -e NVIDIA_VISIBLE_DEVICES="$gpu_uuid" \
    -e VANSMALL_GPU_MEMORY_LIMIT_MB=20000 \
    tensorflow-vansmall-live \
    python -m advanced_visualization.tensorflow_service.extract_features \
      --model-id "$model_id" \
      --csv /app/src/eval/idfraud/annotation/joined_predictions_30june.csv \
      --output "$output" \
      --incremental-from "$output" \
      --image-column "$image_column" \
      --prediction-column "$prediction_column" \
      --batch-size 16
  echo "FEATURE DONE gpu=$gpu model=$model_id $(date -Is)"
}

ensure_crop_alias() {
  python -m advanced_visualization.cli.ensure_csv_alias \
    "$ARTIFACT_ROOT/ench21_vansmall_crop/features/prepared_predictions.csv" \
    --source absolute_ocr_path \
    --target absolute_crop_path
}

gpu0_features() {
  torch_features 0 \
    Ex8point4_UniRepLKNet_B_in22k_legacy_v1_512_crop_epoch7 \
    absolute_ocr_path \
    "$ARTIFACT_ROOT/Ex8point4_UniRepLKNet_B_in22k_legacy_v1_512_crop_epoch7/features/prepared_predictions.csv" \
    16
  torch_features 0 \
    Ex8point2_UniRepLKNet_T_legacy_v1_512_ori_epoch10 \
    absolute_ori_path \
    "$ARTIFACT_ROOT/Ex8point2_UniRepLKNet_T_legacy_v1_512_ori_epoch10/features/prepared_predictions.csv" \
    32
  tensorflow_features 0 ench21_vansmall_crop absolute_crop_path tf_crop_pred \
    "$ARTIFACT_ROOT/ench21_vansmall_crop/features/prepared_predictions.csv"
}

gpu1_features() {
  torch_features 1 \
    InternEnch_Ex8point2res1024largerb \
    absolute_ori_path \
    "$ARTIFACT_ROOT/InternEnch_Ex8point2res1024largerb/features/prepared_predictions.csv" \
    8
  torch_features 1 \
    Ex8point2res1024_moredata_largerbs_UniRepLKNet_T_legacy_v1_1024_ori_epoch11 \
    absolute_ori_path \
    "$ARTIFACT_ROOT/Ex8point2res1024_moredata_largerbs_UniRepLKNet_T_legacy_v1_1024_ori_epoch11/features/prepared_predictions.csv" \
    8
  torch_features 1 \
    square_exp2_Ex8point2res1024_largerbs_21plusaugfeb_UniRepLKNet_T_legacy_v1_1024_ori_epoch8 \
    absolute_ori_path \
    "$ARTIFACT_ROOT/square_exp2_Ex8point2res1024_largerbs_21plusaugfeb_UniRepLKNet_T_legacy_v1_1024_ori_epoch8/features/prepared_predictions.csv" \
    8
  tensorflow_features 1 ench21_vansmall_ori absolute_ori_path tf_ori_pred \
    "$ARTIFACT_ROOT/ench21_vansmall_ori/features/prepared_predictions.csv"
}

cd "$REPO_ROOT"
gpu0_features >"$ARTIFACT_ROOT/june_features_gpu0.log" 2>&1 &
pid0=$!
gpu1_features >"$ARTIFACT_ROOT/june_features_gpu1.log" 2>&1 &
pid1=$!
wait "$pid0"
wait "$pid1"

ensure_crop_alias

python -m advanced_visualization.cli.verify_june_refresh --phase features

python -m advanced_visualization.cli.run_sharded_gradcampp \
  --gpu-slots 0:1 --gpu-slots 1:1 \
  --allocator-limit-mib 22000 --batch-size 8 \
  --model-id Ex8point2_UniRepLKNet_T_legacy_v1_512_ori_epoch10 \
  --model-id Ex8point4_UniRepLKNet_B_in22k_legacy_v1_512_crop_epoch7 \
  --model-id Ex8point2res1024_moredata_largerbs_UniRepLKNet_T_legacy_v1_1024_ori_epoch11 \
  --model-id square_exp2_Ex8point2res1024_largerbs_21plusaugfeb_UniRepLKNet_T_legacy_v1_1024_ori_epoch8 \
  --model-id InternEnch_Ex8point2res1024largerb

python -m advanced_visualization.cli.run_sharded_gradcampp \
  --gpu-slots 0:1 --gpu-slots 1:1 \
  --allocator-limit-mib 20000 --batch-size 1 \
  --model-id ench21_vansmall_ori \
  --model-id ench21_vansmall_crop

AUTOTORCH_IMAGE_CACHE_PATH_ALIASES=/routine_data=/routine_data \
  python -m advanced_visualization.cli.verify_june_refresh --phase all
echo "JUNE REFRESH COMPLETE $(date -Is)"
