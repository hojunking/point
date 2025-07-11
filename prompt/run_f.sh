#!/bin/bash

# Boundary roots (data subset)
BOUNDARY_ROOTS=(
  #"data/boundary/bfa006"
  "data/boundary/pup_scale03_bfa004"
  "data/boundary/pup_scale05_bfa006"
  "data/boundary/pup_scale03_bfa006"
  "data/boundary/pup_scale01_bfa002"
  "data/boundary/pup_scale03_bfa002"
  "data/boundary/pup_scale05_bfa002"
  "data/boundary/pup_scale07_bfa002"
  "data/boundary/pup_scale09_bfa002"
  "data/boundary/pup_scale01_bfa004"
  "data/boundary/pup_scale09_bfa004"
  "data/boundary/pup_scale01_bfa006"
  "data/boundary/pup_scale09_bfa006"
)

# Feature flag 조합들 (밑줄(_)로 구분)
FEATURES_FLAG_COMBOS=(
  "opacity"
  # "scale"
  "scale_opacity"
  # ... 다른 조합들을 여기에 추가 ...
)

for BOUNDARY_ROOT in "${BOUNDARY_ROOTS[@]}"; do
  for COMBO in "${FEATURES_FLAG_COMBOS[@]}"; do
    
    # 실험 이름 설정
    EXP_NAME="ptv3-$(basename "$BOUNDARY_ROOT")_samples100_${COMBO}"

    # --- 옵션 동적 생성 시작 ---
    IN_CHANNELS=6 # 기본 채널 (color 3 + normal 3)
    
    # COMBO 내용에 따라 in_channels 동적 계산
    if [[ "$COMBO" == *"scale"* ]]; then
      IN_CHANNELS=$((IN_CHANNELS + 3))
    fi
    if [[ "$COMBO" == *"opacity"* ]]; then
      IN_CHANNELS=$((IN_CHANNELS + 1))
    fi
    # ... 다른 피처가 있다면 여기에 계산 로직 추가 ...

    # Python이 인식할 수 있는 리스트 형태의 문자열 생성
    # 예: "scale_opacity" -> "'scale','opacity'"
    FEATURES_LIST_CONTENT=$(echo "$COMBO" | sed "s/_/','/g")
    FEATURES_LIST="['$FEATURES_LIST_CONTENT']"

    # 덮어쓸 추가 옵션들을 하나의 변수로 묶기
    # 이 변수는 train.sh에서 환경 변수로 참조됩니다.
    export EXTRA_OPTIONS="model.backbone.in_channels=$IN_CHANNELS"
    export EXTRA_OPTIONS="$EXTRA_OPTIONS data.train.features_flag=\"$FEATURES_LIST\""
    export EXTRA_OPTIONS="$EXTRA_OPTIONS data.val.features_flag=\"$FEATURES_LIST\""
    export EXTRA_OPTIONS="$EXTRA_OPTIONS data.test.features_flag=\"$FEATURES_LIST\""
    
    # train.sh가 사용할 수 있도록 BOUNDARY_ROOT도 환경 변수로 내보내기
    export BOUNDARY_ROOT="$BOUNDARY_ROOT"

    echo "=================================================="
    echo "Launching training for combo: $COMBO"
    echo "Experiment name: $EXP_NAME"
    echo "Overriding with options: $EXTRA_OPTIONS"
    echo "=================================================="

    # train.sh 스크립트 호출
    sh scripts/train.sh \
      -g 1 \
      -d scannet \
      -n "$EXP_NAME" \
      -r false \
      -c semseg-pt-v3m3-bsblock

    # 로그 분석 및 결과 저장
    LOG_PATH="exp/scannet/${EXP_NAME}/train.log"
    python3 ./gspread/gspread_results.py "$LOG_PATH" "$EXP_NAME" sample100_test
  done
done