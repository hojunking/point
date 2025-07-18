#!/bin/bash

# 실험 이름 설정
EXP_NAME="sonata_distill_test"

echo "=================================================="
echo "Experiment name: $EXP_NAME"
echo "=================================================="

# train.sh 스크립트 호출
sh scripts/train.sh \
    -g 1 \
    -d scannet \
    -n "$EXP_NAME" \
    -c semseg-pt-v3m4-bs_distill \

# 로그 분석 및 결과 저장
LOG_PATH="exp/scannet/${EXP_NAME}/train.log"
#python3 ./gspread/gspread_results.py "$LOG_PATH" "$EXP_NAME" sample100_test