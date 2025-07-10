#!/bin/bash

# Boundary roots (data subset)
BOUNDARY_ROOTS=(
  #"data/boundary/bfa006"
  "data/boundary/pup_scale03_bfa004"
  "data/boundary/pup_scale05_bfa006"
  "data/boundary/pup_scale03_bfa006"
  "data/boundary/pup_scale07_bfa006"
)
# Feature flag 조합들 (각각 배열로 정의)
FEATURES_FLAG_COMBOS=(
  "scale"
  "opacity"
  "scale opacity"
)

for BOUNDARY_ROOT in "${BOUNDARY_ROOTS[@]}"; do
  for COMBO in "${FEATURES_FLAG_COMBOS[@]}"; do

    # JSON-like 리스트 문자열로 변환
    FLAGS_ARRAY=($COMBO)  # 공백으로 분리하여 배열로
    FEATURES_FLAG_STR=$(printf '"%s",' "${FLAGS_ARRAY[@]}" | sed 's/,$//')
    FEATURES_FLAG_STR="[$FEATURES_FLAG_STR]"

    # EXP_NAME 설정 (예: scale_opacity 등)
    FLAG_TAG=$(echo "${FLAGS_ARRAY[@]}" | tr ' ' '_')
    EXP_NAME="ptv3-$(basename "$BOUNDARY_ROOT")_samples100_${FLAG_TAG}"

    echo "Launching training for boundary_root: $BOUNDARY_ROOT with features: $FEATURES_FLAG_STR"

    # 환경 변수로 넘기기
    export BOUNDARY_ROOT="$BOUNDARY_ROOT"
    export FEATURES_FLAG="$FEATURES_FLAG_STR"

    sh scripts/train.sh \
      -g 1 \
      -d scannet \
      -n "$EXP_NAME" \
      -r false \
      -c semseg-pt-v3m3-bsblock

    LOG_PATH="exp/scannet/${EXP_NAME}/train.log"
    python3 ./gspread/gspread_results.py "$LOG_PATH" "$EXP_NAME" sample100_test
  done
done
