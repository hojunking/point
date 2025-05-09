#!/bin/bash

DATA_ROOTS=(
  #"data/FPS01_scale"
  # "data/vox004_scale04-low_opacity"
  # "data/vox004_scale06-low_opacity"
  # "data/vox004_scale08-low_opacity"
  "data/scannet"
  #"data/FPS001_opacity"
  #"data/vox006_opacity"
  # "data/FPS005_opacity"
  # "data/vox004_opacity"
  # "data/vox004_scale02_opacity"
  # "data/vox004_scale04_opacity"
  # "data/vox004_scale06_opacity"
  # "data/vox004_scale08_opacity"
  # "data/vox004_scale099_opacity"
)

for ROOT in "${DATA_ROOTS[@]}"
do
  EXP_NAME="scannet-samples100-tf_b-$(basename "$ROOT")-fixs"
  echo "Launching training for data_root: $ROOT"

  DATA_ROOT="$ROOT" \
  sh scripts/train.sh \
    -g 1 \
    -d scannet \
    -n "$EXP_NAME" \
    -r false \
    -c semseg-pt-v3m1-0-base \

  LOG_PATH="exp/scannet/${EXP_NAME}/train.log"
  python3 ./gspread/gspread_results.py "$LOG_PATH" "$EXP_NAME" sample100_test
done
