#!/bin/bash

dataset="cifar-10"
data_path="/data/users/yn621/${dataset}"
model_dir="/data/users/yn621/models/ISO"
device=3

python predict.py --cuda-device $device --dataset $dataset --data-path $data_path