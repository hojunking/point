#!/bin/bash
# Feature flag 조합들 (밑줄(_)로 구분)
FEATURES_FLAG_COMBOS=(
  # 예시) "scale_opacity", "scale_opacity_sh3", "opacity_sh2"
  "scale_opacity"
)

for COMBO in "${FEATURES_FLAG_COMBOS[@]}"; do

# 실험 이름 설정
EXP_NAME="sonata_scale-opacity_k20_400epoch_scannet_dl4"

# --- 옵션 동적 생성 시작 ---
IN_CHANNELS=6 # 기본 채널 (color 3 + normal 3)
SH_DEGREE=0
FEATURE_FLAGS=()

IFS='_' read -ra TOKENS <<< "$COMBO"
for TOKEN in "${TOKENS[@]}"; do
    if [[ "$TOKEN" == sh* ]]; then
        DEGREE_VALUE=${TOKEN#sh}
        if [[ -n "$DEGREE_VALUE" ]]; then
            SH_DEGREE=$DEGREE_VALUE
        else
            SH_DEGREE=3
        fi
        FEATURE_FLAGS+=("sh")
    elif [[ -n "$TOKEN" ]]; then
        FEATURE_FLAGS+=("$TOKEN")
    fi
done

declare -A FEATURE_DIMS=(
    [scale]=3
    [opacity]=1
    [rotation]=3
)

for FLAG in "${FEATURE_FLAGS[@]}"; do
    if [[ "$FLAG" == "sh" ]]; then
        if (( SH_DEGREE > 0 )); then
            SH_DIM=$((3 * (SH_DEGREE + 1) * (SH_DEGREE + 1)))
            IN_CHANNELS=$((IN_CHANNELS + SH_DIM))
        fi
    else
        DIM=${FEATURE_DIMS[$FLAG]}
        if [[ -n "$DIM" ]]; then
            IN_CHANNELS=$((IN_CHANNELS + DIM))
        fi
    fi
done

if (( ${#FEATURE_FLAGS[@]} == 0 )); then
    FEATURES_LIST="[]"
else
    FEATURES_LIST_CONTENT=$(printf "'%s'," "${FEATURE_FLAGS[@]}")
    FEATURES_LIST="[${FEATURES_LIST_CONTENT%,}]"
fi

# 덮어쓸 추가 옵션들을 하나의 변수로 묶기
# 이 변수는 train.sh에서 환경 변수로 참조됩니다.
export EXTRA_OPTIONS="model.backbone.in_channels=$IN_CHANNELS data.train.features_flag=$FEATURES_LIST data.train.sh_degree=$SH_DEGREE"

echo "=================================================="
echo "Launching training for combo: $COMBO"
echo "Experiment name: $EXP_NAME"
echo "Overriding with options: $EXTRA_OPTIONS"
echo "=================================================="

# train.sh 스크립트 호출
sh scripts/train.sh \
    -g 1 \
    -d sonata \
    -n "$EXP_NAME" \
    -c pretrain-sonata-v1m1-0-base \
    -r true
    #-c semseg-sonata-v1m1-0c-scannet-ft \
    #-w pre_trained/sonata_opacity_k30_800epoch_loss051.pth

# 로그 분석 및 결과 저장
LOG_PATH="exp/sonata/${EXP_NAME}/train.log"
python3 ./gspread/gspread_results.py "$LOG_PATH" "$EXP_NAME" sample100_test
done
