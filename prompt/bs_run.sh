#!/bin/bash

DATA_ROOTS=(
  "data/scannet"
)

for ROOT in "${DATA_ROOTS[@]}"
do
  EXP_NAME="$(basename "$ROOT")-samples_ptv3-bs_b05"
  echo "Launching training for data_root: $ROOT"

  DATA_ROOT="$ROOT" \
  sh scripts/train.sh \
    -g 1 \
    -d scannet \
    -n "$EXP_NAME" \
    -r false \
    -c semseg-pt-v3m3-bsblock \

  LOG_PATH="exp/scannet/${EXP_NAME}/train.log"
 #python3 ./gspread/gspread_results.py "$LOG_PATH" "$EXP_NAME" sample100_test
done
