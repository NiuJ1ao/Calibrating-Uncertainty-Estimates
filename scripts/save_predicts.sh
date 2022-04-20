#!/bin/bash

date="$(date +'%d-%m-%Y')"
mkdir -p "log/$date"

seeds=(864 394 776 911 430)
devices=(0 0 0 0 0)
# seeds=(864)
# devices=(0)

model="resnet"
model_dir="/mnt/e/models/iso_models/"
dataset="SVHN"
data_path="/mnt/e/data/${dataset}"

# for i in "${!seeds[@]}"; do
#     echo "seeds=${seeds[i]}, device=${devices[i]}"

#     python save_predicts.py \
#     --seed ${seeds[i]} --cuda-device ${devices[i]} \
#     --model $model --model-dir $model_dir \
#     --dataset $dataset --data-path $data_path
# done

python save_predicts.py \
--cuda-device 0 --ensemble \
--model $model --model-dir $model_dir \
--dataset $dataset --data-path $data_path
