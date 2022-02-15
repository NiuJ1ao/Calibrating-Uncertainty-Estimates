#!/bin/bash


seeds=(864 394 776 911 430)
devices=(0 1 2 3)
model="resnet"
dataset="cifar10"

for i in "${!devices[@]}"; do
    echo "seeds=${seeds[i]}, device=${devices[i]}"
    log_name="log/${model}_${dataset}_${seeds[i]}_${devices[i]}.log"

    nohup python main.py --seed ${seeds[i]} --cuda-device ${devices[i]} --num-epochs 1000 --dataset $dataset > $log_name 2>&1 &
done
