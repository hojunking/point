DATA_ROOTS=(
# "data/boundary/mah_k20_scale07_bfa004_v2"
# "data/boundary/mah_k20_scale05_bfa004_v2"
"data/boundary/mah_k20_scale07_bfa004_v2"
)

for ROOT in "${DATA_ROOTS[@]}"
do
  EXP_NAME="insg_$(basename "$ROOT")_bs03-lr6_local"
  EXTRA_OPTIONS="boundary_root=$ROOT data.train.boundary_root=$ROOT data.val.boundary_root=$ROOT data.test.boundary_root=$ROOT"
  echo "Launching training for boundary_root: $ROOT"

  EXTRA_OPTIONS="$EXTRA_OPTIONS" \
  sh scripts/train.sh \
    -g 1 \
    -d scannet \
    -n "$EXP_NAME" \
    -r false \
    -c insseg-pointgroup-v1m3-0-ptv3-bsblock

  LOG_PATH="exp/scannet/${EXP_NAME}/train.log"
  python3 ./gspread/gspread_results_insseg.py "$LOG_PATH" "$EXP_NAME" sample100_test
done


 
