#!/bin/bash


seeds=(864 394 776 911 430)
devices=(1 1 3 3)
model="resnet"
dataset="SVHN"
data_path="/data/users/yn621/${dataset}"
epochs=1000

for i in "${!devices[@]}"; do
    echo "seeds=${seeds[i]}, device=${devices[i]}"
    log_name="log/${model}_${dataset}_${seeds[i]}_${devices[i]}.log"

    nohup python train.py --seed ${seeds[i]} --cuda-device ${devices[i]} --num-epochs $epochs --dataset $dataset --data-path $data_path > $log_name 2>&1 &
done

# nohup python train.py --seed 430 --cuda-device 1 --num-epochs $epochs --dataset cifar-10 --data-path /data/users/yn621/cifar-10 > log/${model}_cifar-10_430_1.log 2>&1 &

# nohup python train.py --seed 430 --cuda-device 1 --num-epochs $epochs --dataset SVHN --data-path /data/users/yn621/SVHN > log/${model}_SVHN_430_1.log 2>&1 &