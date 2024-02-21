#!/bin/bash

MODEL_NAME="MLP_High_Variance"
DATA_TYPE="v2_231111"
DEVICE="2"
BATCH_SIZE=4096
SEEDS=(2048)
SPLIT_BY=("drug" "cell")

for SEED in "${SEEDS[@]}"
do
    for SPLIT in "${SPLIT_BY[@]}"
    do
        echo "Running ${MODEL_NAME}_v2.py with SPLIT_BY=$SPLIT and SEED=$SEED"

        python MLP_High_Variance_v2.py \
            --model_name="${MODEL_NAME}_${SEED}_${DATA_TYPE}" \
            --device="$DEVICE" \
            --split_by="$SPLIT" \
            --batch_size="$BATCH_SIZE" \
            --seed="$SEED"

        echo "Finished running ${MODEL_NAME}_v2.py with SPLIT_BY=$SPLIT and SEED=$SEED"
    done
done
