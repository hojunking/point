#!/bin/bash

DATA_ROOTS=(
  "data/boundary/b_s07-o03"
)

for ROOT in "${DATA_ROOTS[@]}"
do
  EXP_NAME="ptv3-$(basename "$ROOT")-samples"
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
