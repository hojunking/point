#!/bin/bash

DATA_ROOTS=(
# "data/boundary/mah_k20_scale07_bfa004_v2"
# "data/boundary/mah_k20_scale05_bfa004_v2"
"data/boundary/mah_scale01"
"data/boundary/mah_scale09"
)

for ROOT in "${DATA_ROOTS[@]}"
do
  EXP_NAME="octformer-$(basename "$ROOT")_bs-lr2_local"
  EXTRA_OPTIONS="boundary_root=$ROOT data.train.boundary_root=$ROOT data.val.boundary_root=$ROOT data.test.boundary_root=$ROOT"
  echo "Launching OctFormer-BSBlock training for boundary_root: $ROOT"

  EXTRA_OPTIONS="$EXTRA_OPTIONS" \
  sh scripts/train.sh \
    -g 1 \
    -d scannet \
    -n "$EXP_NAME" \
    -r false \
    -c semseg-octformer-v1m2-bsblock

  LOG_PATH="exp/scannet/${EXP_NAME}/train.log"
  python3 ./gspread/gspread_results.py "$LOG_PATH" "$EXP_NAME" sample100_test
done
