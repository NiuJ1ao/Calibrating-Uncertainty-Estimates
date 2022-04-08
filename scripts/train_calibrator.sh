#!/bin/bash

date="$(date +'%d-%m-%Y')"
mkdir -p "log/$date"
seeds=(864 394 776 911 430)
devices=(0 0 0 0 0)
model="resnet"
calibrate="temp"
model_dir="/mnt/e/models/iso_models/"
dataset="SVHN"
data_path="/mnt/e/data/${dataset}"
epochs=100
lr=1e-2

for i in "${!seeds[@]}"; do
    echo "seeds=${seeds[i]}, device=${devices[i]}"
    log_name="log/$date/${model}-${calibrate}_${dataset}_${seeds[i]}_${devices[i]}.log"

    python train_calibrator.py \
    --seed ${seeds[i]} --cuda-device ${devices[i]} \
    --model $model --calibrate $calibrate --model-dir $model_dir \
    --num-epochs $epochs --lr $lr \
    --dataset $dataset --data-path $data_path \
    # > $log_name 2>&1
done

# seed=864
# device=0

# python train_calibrator.py \
# --seed $seed --cuda-device $device \
# --model $model --calibrate $calibrate --model-dir $model_dir \
# --num-epochs $epochs --lr $lr \
# --dataset $dataset --data-path $data_path