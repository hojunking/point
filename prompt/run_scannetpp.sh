#!/bin/bash

DATA_ROOTS=(
  "data/scannetpp"
)

for ROOT in "${DATA_ROOTS[@]}"
do
  EXP_NAME="ptv3_baseline-$(basename "$ROOT")_lr4_dl4-t2"
  echo "Launching training for data_root: $ROOT"

  DATA_ROOT="$ROOT" \
  sh scripts/train.sh \
    -g 1 \
    -d scannetpp \
    -n "$EXP_NAME" \
    -r true \
    -c semseg-pt-v3m1-0-base \

  LOG_PATH="exp/scannetpp/${EXP_NAME}/train.log"
  python3 ./gspread/gspread_results.py "$LOG_PATH" "$EXP_NAME" sample100_test
done




# BOUNDARY_ROOTS=(
#   "data/boundary/scannetpp_scale07_bfa004"
# )

# for BOUNDARY_ROOT in "${BOUNDARY_ROOTS[@]}"; do
    
#   # 실험 이름 설정
#   EXP_NAME="$(basename "$BOUNDARY_ROOT")_bs09_distill04-lr1_dl4"

#   # train.sh가 사용할 수 있도록 BOUNDARY_ROOT도 환경 변수로 내보내기
#   export BOUNDARY_ROOT="$BOUNDARY_ROOT"

#   echo "=================================================="
#   echo "Experiment name: $EXP_NAME"
#   echo "Overriding with options: $EXTRA_OPTIONS"
#   echo "=================================================="

#   # train.sh 스크립트 호출
#   sh scripts/train.sh \
#     -g 1 \
#     -d scannetpp \
#     -n "$EXP_NAME" \
#     -r true \
#     -c semseg-pt-v3m1-2-bs_distill

#   # 로그 분석 및 결과 저장
#   LOG_PATH="exp/scannetpp/${EXP_NAME}/train.log"
#   python3 ./gspread/gspread_results.py "$LOG_PATH" "$EXP_NAME" sample100_test
# done