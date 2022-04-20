#!/bin/bash

date="$(date +'%d-%m-%Y')"
mkdir -p "log/$date"

# seeds=(864 394 776 911 430)
# devices=(0 0 0 0 0)
seeds=864
devices=0

model="resnet"
calibrate="platt"
model_dir="/mnt/e/models/iso_models/"
dataset="SVHN"
data_path="/mnt/e/data/${dataset}"
epochs=100
lr=1e-2

python train_calibrator.py \
--seed 864 --cuda-device 0 \
--model $model --calibrate $calibrate --model-dir $model_dir \
--num-epochs $epochs --lr $lr \
--dataset $dataset --data-path $data_path \

device=0
python train_calibrator.py \
--ensemble --cuda-device $device \
--model $model --calibrate $calibrate --model-dir $model_dir \
--num-epochs $epochs --lr $lr \
--dataset $dataset --data-path $data_path