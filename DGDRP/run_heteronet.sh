#!/bin/bash

CURRENT_DIR="/data/project/inyoung/DGDRP"
DATA_DIR="${CURRENT_DIR}/Data"
GRAPH_ROOT_DIR="${DATA_DIR}/graph_pyg/20_indirect_targets"

MODEL_NAME="DGDRP_Top1000"
DATA_TYPE="20_indirect_targets"
SPLIT_BY=("drug") # ("drug" "cell")
DEVICE="2"
BATCH_SIZE=64
SEEDS=(1024 1234 4567)

for SEED in "${SEEDS[@]}"
do
    for SPLIT in "${SPLIT_BY[@]}"
    do
        echo "Running heteronet.py with SEED=$SEED"

        python heteronet.py \
            --data_type="${DATA_TYPE}" \
            --currentdir="${CURRENT_DIR}" \
            --datadir="${DATA_DIR}" \
            --model_name="${MODEL_NAME}_${SEED}_${DATA_TYPE}" \
            --root="${GRAPH_ROOT_DIR}" \
            --device="${DEVICE}" \
            --split_by="${SPLIT_BY}" \
            --batch_size="${BATCH_SIZE}" \
            --seed="${SEED}"

        echo "Finished running heteronet.py with SEED=$SEED"
    done
done