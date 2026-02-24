#!/bin/bash
# baseline
# DATA_ROOTS=(
#   #"data/FPS01_scale"
#   "data/scannet"
# )

# for ROOT in "${DATA_ROOTS[@]}"
# do
#   EXP_NAME="scannet_oct_b-scannet"
#   echo "Launching training for data_root: $ROOT"

#   DATA_ROOT="$ROOT" \
#   sh scripts/train.sh \
#     -g 1 \
#     -d scannet \
#     -n "$EXP_NAME" \
#     -r true \
#     -c semseg-octformer-v1m1-0-base \

#   LOG_PATH="exp/scannet/${EXP_NAME}/train.log"
#   python3 ./gspread/gspread_results.py "$LOG_PATH" "$EXP_NAME" sample100_test
# done

## scannet v2
# DATA_ROOTS=(
# # "data/boundary/mah_k20_scale07_bfa004_v2"
# # "data/boundary/mah_k20_scale05_bfa004_v2"
# "data/boundary/bfa002"
# )

# for ROOT in "${DATA_ROOTS[@]}"
# do
#   EXP_NAME="$(basename "$ROOT")_baseline"
#   EXTRA_OPTIONS="boundary_root=$ROOT data.train.boundary_root=$ROOT data.val.boundary_root=$ROOT data.test.boundary_root=$ROOT"
#   echo "Launching training for boundary_root: $ROOT"

#   EXTRA_OPTIONS="$EXTRA_OPTIONS" \
#   sh scripts/train.sh \
#     -g 1 \
#     -d scannet \
#     -n "$EXP_NAME" \
#     -r false \
#     -c semseg-octformer-v1m2-bfa-bs

#   LOG_PATH="exp/scannet/${EXP_NAME}/train.log"
#   python3 ./gspread/gspread_results.py "$LOG_PATH" "$EXP_NAME" sample100_test
# done

#!/bin/bash

# Boundary roots (data subset)
BOUNDARY_ROOTS=(
  #"data/boundary/scenesplats_scale03_bfa004"
  #"data/boundary/scenesplats_scale05_bfa004"
"data/boundary/bfa002"

)

for BOUNDARY_ROOT in "${BOUNDARY_ROOTS[@]}"; do
    
  # 실험 이름 설정
  EXP_NAME="scannet200-$(basename "$BOUNDARY_ROOT")_baseline"

  # config override 옵션 구성 (boundary_root를 train/val/test 모두에 주입)
  EXTRA_OPTIONS="boundary_root=$BOUNDARY_ROOT data.train.boundary_root=$BOUNDARY_ROOT data.val.boundary_root=$BOUNDARY_ROOT data.test.boundary_root=$BOUNDARY_ROOT"

  echo "=================================================="
  echo "Experiment name: $EXP_NAME"
  echo "Overriding with options: $EXTRA_OPTIONS"
  echo "=================================================="

  # train.sh 스크립트 호출
  EXTRA_OPTIONS="$EXTRA_OPTIONS" \
  sh scripts/train.sh \
    -g 1 \
    -d scannet200 \
    -n "$EXP_NAME" \
    -r false \
    -c semseg-octformer-v1m2-bfa-bs

  # 로그 분석 및 결과 저장
  LOG_PATH="exp/scannet200/${EXP_NAME}/train.log"
  python3 ./gspread/gspread_results.py "$LOG_PATH" "$EXP_NAME" sample100_test
done
