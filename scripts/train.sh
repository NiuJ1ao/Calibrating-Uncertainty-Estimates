#!/bin/bash

date="$(date +'%d-%m-%Y')"
seeds=(864 394 776 911 430)
devices=(0 0 0 0 0)
model="resnet"
dataset="cifar-10"
data_path="/data/users/yn621/${dataset}"
epochs=1000

for i in "${!devices[@]}"; do
    echo "seeds=${seeds[i]}, device=${devices[i]}"
    log_name="log/${model}_${dataset}_${seeds[i]}_${devices[i]}.log"

    nohup python train.py --seed ${seeds[i]} --cuda-device ${devices[i]} --num-epochs $epochs --dataset $dataset --data-path $data_path > $log_name 2>&1 &
done

# model="resnet"
# dataset="cifar-10"
# data_path="/data/users/yn621/${dataset}"
# seed=911
# device=3
# epochs=1000
# log_name="log/$date/${model}_${dataset}_${seed}_${device}.log"
# mkdir -p "log/$date"


# nohup python train.py --seed $seed --cuda-device $device --num-epochs $epochs --dataset $dataset --data-path $data_path > $log_name 2>&1 &
# tail -f $log_name

# nohup python train.py --seed 430 --cuda-device 1 --num-epochs $epochs --dataset SVHN --data-path /data/users/yn621/SVHN > log/${model}_SVHN_430_1.log 2>&1 &