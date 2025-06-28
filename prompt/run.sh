#!/bin/bash

DATA_ROOTS=(
  #"data/FPS01_scale"
#  "data/scannet"
  "data/features/base_3dgs"

  # "data/vox004_scale-rotation-opacity"
)

for ROOT in "${DATA_ROOTS[@]}"
do
  EXP_NAME="scannet_tf_b-$(basename "$ROOT")"
  echo "Launching training for data_root: $ROOT"

  DATA_ROOT="$ROOT" \
  sh scripts/train.sh \
    -g 1 \
    -d scannet \
    -n "$EXP_NAME" \
    -r false \
    -c semseg-pt-v3m1-0-base \

  LOG_PATH="exp/scannet/${EXP_NAME}/train.log"
  #python3 ./gspread/gspread_results.py "$LOG_PATH" "$EXP_NAME" sample100_test
done
