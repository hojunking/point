#!/bin/bash

DATA_ROOTS=(
#  "data/scannet"
  "data/matterport3d"
)

for ROOT in "${DATA_ROOTS[@]}"
do
  EXP_NAME="ptv3_baseline-$(basename "$ROOT")_lr2_local"
  echo "Launching training for data_root: $ROOT"

  DATA_ROOT="$ROOT" \
  sh scripts/train.sh \
    -g 1 \
    -d matterport3d \
    -n "$EXP_NAME" \
    -r true \
    -c semseg-pt-v3m1-0-base \

  LOG_PATH="exp/scannet/${EXP_NAME}/train.log"
  python3 ./gspread/gspread_results.py "$LOG_PATH" "$EXP_NAME" sample100_test
done
