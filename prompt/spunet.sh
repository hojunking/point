#!/bin/bash

DATA_ROOTS=(
   #"data/3dgs_pdistance0005_pruned"
  "data/vox004_scale04-low_opacity"
   #"data/scale075"
   #"data/3dgs_pdistance00008_pruned"
)

for ROOT in "${DATA_ROOTS[@]}"
do
  EXP_NAME="scannet-samples-spunet-b-$(basename "$ROOT")"
  echo "Launching training for data_root: $ROOT"

  DATA_ROOT="$ROOT" \
  sh scripts/train.sh \
    -g 1 \
    -d scannet \
    -n "$EXP_NAME" \
    -r false \
    -c semseg-spunet-v1m1-0-base \

  LOG_PATH="exp/scannet/${EXP_NAME}/train.log"
  python3 ./gspread/gspread_results.py "$LOG_PATH" "$EXP_NAME" sample100_test
done
