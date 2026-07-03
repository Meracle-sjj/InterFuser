#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
GPU_NUM=1
DATASET_ROOT='/home/shijj/interfuser/dataset_collection'

# Dataset Info updated based on scan:
# Weathers available: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 20
# Training Split: Towns 1, 4, 5
# Validation Split: Town 3

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR/.."

./distributed_train.sh $GPU_NUM $DATASET_ROOT  --dataset carla --train-towns 1 4 5 --val-towns 3 \
    --train-weathers 0 1 2 3 4 5 6 7 8 9 10 11 12 13 15 20 --val-weathers 0 1 2 3 4 5 6 7 8 9 10 11 12 13 15 20 \
    --model interfuser_baseline --sched cosine --epochs 25 --warmup-epochs 5 --lr 0.0005 --batch-size 16  -j 16 --no-prefetcher --eval-metric l1_error \
    --opt adamw --opt-eps 1e-8 --weight-decay 0.05  \
    --scale 0.9 1.1 --saver-decreasing --clip-grad 10 --freeze-num -1 \
    --with-backbone-lr --backbone-lr 0.0002 \
    --multi-view --with-lidar --multi-view-input-size 3 128 128 \
    --experiment interfuser_baseline_new_dataset \
    --pretrained

