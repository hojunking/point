#!/bin/bash
# Feature flag 조합들 (밑줄(_)로 구분)
FEATURES_FLAG_COMBOS=(
  # 예시) "scale_opacity", "opacity", "rotation_sh1"
  "scale"
  "scale_opacity"
)

declare -A TEACHER_CKPT_BY_COMBO=(
  ["scale"]="pre_trained/sonata_scale_k20_400epoch_scannet.pth"
  ["scale_opacity"]="pre_trained/REPLACE_WITH_SCALE_OPACITY_TEACHER.pth"
)

for COMBO in "${FEATURES_FLAG_COMBOS[@]}"; do

  # --- 옵션 동적 생성 시작 ---
  SH_DEGREE=0
  FEATURE_FLAGS=()
  FEATURES_DIM=0

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
              if (( SH_DIM > 48 )); then
                  SH_DIM=48
              fi
              FEATURES_DIM=$((FEATURES_DIM + SH_DIM))
          fi
      else
          DIM=${FEATURE_DIMS[$FLAG]}
          if [[ -n "$DIM" ]]; then
              FEATURES_DIM=$((FEATURES_DIM + DIM))
          fi
      fi
  done

  if (( ${#FEATURE_FLAGS[@]} == 0 )); then
      FEATURES_LIST="[]"
  else
      FEATURES_LIST_CONTENT=$(printf "'%s'," "${FEATURE_FLAGS[@]}")
      FEATURES_LIST="[${FEATURES_LIST_CONTENT%,}]"
  fi

  TEACHER_IN_CHANNELS=$((3 + FEATURES_DIM)) # color(3) + GS features
  EXP_NAME="bs09_distill04_${COMBO}_lr2_local"
  TEACHER_CKPT="${TEACHER_CKPT_BY_COMBO[$COMBO]}"

  export EXTRA_OPTIONS="model.teacher_backbone.in_channels=$TEACHER_IN_CHANNELS"
  EXTRA_OPTIONS="$EXTRA_OPTIONS model.teacher_backbone.checkpoint_path=$TEACHER_CKPT"
  EXTRA_OPTIONS="$EXTRA_OPTIONS data.train.features_flag=$FEATURES_LIST data.val.features_flag=$FEATURES_LIST data.test.features_flag=$FEATURES_LIST"
  EXTRA_OPTIONS="$EXTRA_OPTIONS data.train.sh_degree=$SH_DEGREE data.val.sh_degree=$SH_DEGREE data.test.sh_degree=$SH_DEGREE"

  echo "=================================================="
  echo "Experiment name: $EXP_NAME"
  echo "Overriding with options: $EXTRA_OPTIONS"
  echo "=================================================="

  # train.sh 스크립트 호출
  sh scripts/train.sh \
    -g 1 \
    -d scannet \
    -n "$EXP_NAME" \
    -r false \
    -c semseg-pt-v3m5-bs_distill

  # 로그 분석 및 결과 저장
  LOG_PATH="exp/scannet/${EXP_NAME}/train.log"
  python3 ./gspread/gspread_results.py "$LOG_PATH" "$EXP_NAME" sample100_test
done


