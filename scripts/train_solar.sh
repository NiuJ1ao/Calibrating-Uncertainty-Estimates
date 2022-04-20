#!/bin/bash

seeds=(864 394 776 911 430)
devices=(0 0 0 0 0)
# seeds=(864)
# devices=(0)

model="mlp"
model_dir="/mnt/e/models/iso_models"
dataset="solar"
data_path="/mnt/e/data/${dataset}"
epochs=5000
lr=0.001
dropout=0.01
hid_size=40

date="$(date +'%d-%m-%Y')"
mkdir -p "log/$date"

for i in "${!devices[@]}"; do
    echo "seed=${seeds[i]}, device=${devices[i]}"
    log_name="log/$date/${model}_${dataset}_${seeds[i]}_${devices[i]}.log"

    python train_regressor.py \
    --seed ${seeds[i]} --cuda-device ${devices[i]} \
    --num-epochs $epochs --lr $lr --dropout $dropout --hidden-size $hid_size \
    --dataset $dataset --data-path $data_path \
    --model $model --model-dir $model_dir \
    > $log_name 2>&1
done