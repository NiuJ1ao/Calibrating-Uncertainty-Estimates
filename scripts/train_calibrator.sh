#!/bin/bash


# seeds=(864 394 776 911 430)
# devices=(1 1 3 3)
seeds=(864)
devices=(2)
model="resnet-platt"
dataset="cifar-10"
data_path="/data/users/yn621/${dataset}"
epochs=10
lr=0.1

for i in "${!devices[@]}"; do
    echo "seeds=${seeds[i]}, device=${devices[i]}"
    log_name="log/${model}_${dataset}_${seeds[i]}_${devices[i]}.log"

    python train_calibrator.py \
    --seed ${seeds[i]} --cuda-device ${devices[i]} \
    --num-epochs $epochs --lr $lr \
    --dataset $dataset --data-path $data_path
done