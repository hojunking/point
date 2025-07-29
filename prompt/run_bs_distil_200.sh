#!/bin/bash

# Boundary roots (data subset)
BOUNDARY_ROOTS=(
  #"data/boundary/scenesplats_scale03_bfa004"
  #"data/boundary/scenesplats_scale05_bfa004"
  "data/boundary/scenesplats_scale07_bfa004"
)

for BOUNDARY_ROOT in "${BOUNDARY_ROOTS[@]}"; do
    
  # 실험 이름 설정
  EXP_NAME="ptv3-$(basename "$BOUNDARY_ROOT")_bs08-distill02-01_lr6_epoch800"

  # train.sh가 사용할 수 있도록 BOUNDARY_ROOT도 환경 변수로 내보내기
  export BOUNDARY_ROOT="$BOUNDARY_ROOT"

  echo "=================================================="
  echo "Experiment name: $EXP_NAME"
  echo "Overriding with options: $EXTRA_OPTIONS"
  echo "=================================================="

  # train.sh 스크립트 호출
  sh scripts/train.sh \
    -g 1 \
    -d scannet200 \
    -n "$EXP_NAME" \
    -r false \
    -c semseg-pt-v3m1-2-bs_distill

  # 로그 분석 및 결과 저장
  LOG_PATH="exp/scannet/${EXP_NAME}/train.log"
  python3 ./gspread/gspread_results.py "$LOG_PATH" "$EXP_NAME" sample100_test
done