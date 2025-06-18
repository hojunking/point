#!/bin/bash

DATA_ROOTS=(
  #"data/FPS01_scale"
  "data/scannet"
#   "data/vox004_rotation04_opacity"
#   "data/vox004_scale06_opacity"
#   "data/vox004_opacity"
  #"data/vox004_opacity06_scale-rotation-opacity"
  
  # "data/vox004_rotation04_scale-rotation-opacity"
  # "data/vox004_scale-rotation"
  # "data/vox004_rotation-opacity"
  # "data/vox004_scale-rotation-opacity"

)

for ROOT in "${DATA_ROOTS[@]}"
do
  EXP_NAME="scannet_oct_b-$(basename "$ROOT")"
  echo "Launching training for data_root: $ROOT"

  DATA_ROOT="$ROOT" \
  sh scripts/train.sh \
    -g 1 \
    -d scannet \
    -n "$EXP_NAME" \
    -r true \
    -c semseg-octformer-v1m1-0-base \

  LOG_PATH="exp/scannet/${EXP_NAME}/train.log"
  python3 ./gspread/gspread_results.py "$LOG_PATH" "$EXP_NAME" sample100_test
done
