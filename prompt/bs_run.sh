#!/bin/bash

DATA_ROOTS=(
#  "data/boundary/bs_s6-o2"
#"data/boundary/b_s07-o03"
#"data/boundary/bfa_label"
#"data/boundary/bfa"
"data/boundary/bfa06-gs01-gb02-pr5_label01"
"data/boundary/bfa04-gs01-gb02-pr5_label01"
"data/boundary/bfa06-gs01-gb02-pr4_label01"
"data/boundary/bfa06-gs01-gb02-pr6_label01"
"data/boundary/bfa08-gs01-gb02-pr5_label01"
"data/boundary/bfa08-gs01-gb02-pr8_label01"
)

for ROOT in "${DATA_ROOTS[@]}"
do
  EXP_NAME="ptv3-$(basename "$ROOT")_t2"
  echo "Launching training for data_root: $ROOT"

  DATA_ROOT="$ROOT" \
  sh scripts/train.sh \
    -g 1 \
    -d scannet \
    -n "$EXP_NAME" \
    -r false \
    -c semseg-pt-v3m3-bsblock \

  LOG_PATH="exp/scannet/${EXP_NAME}/train.log"
 python3 ./gspread/gspread_results.py "$LOG_PATH" "$EXP_NAME" sample100_test
done
