#!/bin/bash

MODEL_NAME="HeteroNet_Top1000"
DATA_TYPE="20_indirect_targets"
ADJ_FNAME="template_adjacency_matrix_${DATA_TYPE}.tsv"
SPLIT_BY="drug"
DEVICE="0"
SEEDS=(1024 1234 4567)

for SEED in "${SEEDS[@]}"
do
    echo "Running heteronet.py with SEED=$SEED"

    python heteronet.py \
        --model_name="${MODEL_NAME}_${SEED}_${DATA_TYPE}" \
        --template_adj_fname="$ADJ_FNAME" \
        --device="$DEVICE" \
        --split_by="$SPLIT_BY" \
        --seed="$SEED"

    echo "Finished running heteronet.py with SEED=$SEED"
done
