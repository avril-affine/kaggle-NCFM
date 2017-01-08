#!/bin/bash
DATA_DIR="/home/panda/Desktop/Projects/fish/data/train_folds"

for i in `seq 0 9`;
do
    echo "------------------Training Fold $i------------------"
    python train.py $DATA_DIR $i
done
