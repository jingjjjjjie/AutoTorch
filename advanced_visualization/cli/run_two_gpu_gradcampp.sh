#!/usr/bin/env bash

set -uo pipefail

REPO_ROOT="/home/jingjie/AutoTorch"
ROOT="/mnt4/advanced_visualization"
COMPOSE_FILE="$REPO_ROOT/advanced_visualization/docker-compose.yml"
STATE="$ROOT/gradcam_multigpu_state.json"

gpu0_models=(
  Ex8point2_UniRepLKNet_T_legacy_v1_512_ori_epoch10
  square_exp2_Ex8point2res1024_largerbs_21plusaugfeb_UniRepLKNet_T_legacy_v1_1024_ori_epoch8
  ench21_vansmall_ori
)

gpu1_models=(
  Ex8point4_UniRepLKNet_B_in22k_legacy_v1_512_crop_epoch7
  Ex8point2res1024_moredata_largerbs_UniRepLKNet_T_legacy_v1_1024_ori_epoch11
  ench21_vansmall_crop
)

model_args() {
  local model
  for model in "$@"; do
    printf '%s\n' --model-id "$model"
  done
}

clean_artifacts() {
  echo "CLEAN START $(date -Is)"
  docker compose -f "$COMPOSE_FILE" run --rm --no-deps \
    --entrypoint /bin/sh tensorflow-vansmall-live -c \
    "find '$ROOT' -type f \( -name '*.webp' -o -name 'gradcam_generation_state*.json' -o -name 'gradcam_run_state.json' \) -delete"
  local remaining
  remaining=$(find "$ROOT" -type f -name '*.webp' | wc -l)
  if [[ "$remaining" -ne 0 ]]; then
    echo "CLEAN FAILED remaining_webp=$remaining"
    return 1
  fi
  echo "CLEAN DONE $(date -Is)"
}

run_worker() {
  local gpu=$1
  local phase=$2
  local ready_used_mib=$3
  local memory_limit_mib=$4
  shift 4
  local models=("$@")
  local args=()
  local item
  while IFS= read -r item; do
    args+=("$item")
  done < <(model_args "${models[@]}")

  PYTHONUNBUFFERED=1 \
  CUDA_DEVICE_ORDER=PCI_BUS_ID \
  CUDA_VISIBLE_DEVICES="$gpu" \
  NVIDIA_VISIBLE_DEVICES="$gpu" \
  python -m advanced_visualization.cli.run_registered_gradcampp \
    --gpu "$gpu" \
    --phase "$phase" \
    "${args[@]}" \
    --memory-limit-mib "$memory_limit_mib" \
    --allocator-limit-mib 12000 \
    --wait-for-gpu \
    --ready-used-mib "$ready_used_mib" \
    --wait-interval 60
}

run_worker_until_success() {
  local attempt=1
  local rc
  while true; do
    echo "WORKER START gpu=$1 phase=$2 attempt=$attempt $(date -Is)"
    if run_worker "$@"; then
      echo "WORKER DONE gpu=$1 phase=$2 attempt=$attempt $(date -Is)"
      return 0
    else
      rc=$?
      echo "WORKER RETRY gpu=$1 phase=$2 attempt=$attempt rc=$rc $(date -Is)"
    fi
    attempt=$((attempt + 1))
    sleep 60
  done
}

run_phase() {
  local phase=$1
  local rc0=0
  local rc1=0

  echo "START phase=$phase $(date -Is)"
  run_worker_until_success 0 "$phase" 3000 20000 "${gpu0_models[@]}" \
    >"$ROOT/gpu0_${phase}.log" 2>&1 &
  local pid0=$!
  run_worker_until_success 1 "$phase" 10000 15000 "${gpu1_models[@]}" \
    >"$ROOT/gpu1_${phase}.log" 2>&1 &
  local pid1=$!

  wait "$pid0" || rc0=$?
  wait "$pid1" || rc1=$?
  echo "DONE phase=$phase gpu0_rc=$rc0 gpu1_rc=$rc1 $(date -Is)"

  if [[ "$rc0" -ne 0 || "$rc1" -ne 0 ]]; then
    printf '{"status":"failed","phase":"%s","gpu0_rc":%s,"gpu1_rc":%s}\n' \
      "$phase" "$rc0" "$rc1" >"$STATE"
    return 1
  fi
}

cd "$REPO_ROOT" || exit 1
if ! clean_artifacts; then
  printf '{"status":"failed","phase":"cleanup"}\n' >"$STATE"
  exit 1
fi

printf '{"status":"running","phase":"final","gpus":[0,1]}\n' >"$STATE"
if ! run_phase final; then
  exit 1
fi

printf '{"status":"complete","phase":"final","gpus":[0,1],"method":"gradcam++"}\n' >"$STATE"
echo "ALL COMPLETE $(date -Is)"
