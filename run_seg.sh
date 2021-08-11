#!/bin/sh

python run.py \
    --task segmentation \
    --encoder deeplab50 \
    --batch-size 6 \
    --epochs 50 \
    --learning-rate 1e-3 \
    --num-classes 19 \
    --exp-name CSG-lightning-segmentation \
    --num-patches 8 \
    --moco-weight 1 \
    --moco-queue-size 49152 \
    $@