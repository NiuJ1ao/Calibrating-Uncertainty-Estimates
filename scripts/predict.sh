#!/bin/bash

model="resnet"
calibrate="platt"
dataset="cifar-10"
data_path="/mnt/e/data/${dataset}"
model_dir="/mnt/e/models/iso_models"
device=0

python predict.py \
--model $model --model-dir $model_dir --cuda-device $device \
--dataset $dataset --data-path $data_path --calibrate $calibrate \
--ensemble
