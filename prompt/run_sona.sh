#!/bin/bash

EXP_NAME="sonata_pupgs10_400epoch_scannet_dl5"
CONFIG="pretrain-sonata-v1m1-0-opacity"
export EXTRA_OPTIONS=""

echo "=================================================="
echo "Experiment name: $EXP_NAME"
echo "Config: $CONFIG"
echo "=================================================="

sh scripts/train.sh \
    -g 1 \
    -d sonata \
    -n "$EXP_NAME" \
    -c "$CONFIG" \
    -r false
    #-c semseg-sonata-v1m1-0c-scannet-ft \
    #-w pre_trained/sonata_opacity_k30_800epoch_loss051.pth

LOG_PATH="exp/sonata/${EXP_NAME}/train.log"
python3 ./gspread/gspread_results.py "$LOG_PATH" "$EXP_NAME" sample100_test
