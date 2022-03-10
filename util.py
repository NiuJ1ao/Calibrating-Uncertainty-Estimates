import argparse
import random
import numpy as np
import torch

def args_parser():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--data-path", type=str, default="/data/users/yn621/cifar-10", help="The root path to data directory")
    parser.add_argument("--dataset", type=str, default="cifar-10", choices=["cifar-10", "SVHN"], help="")
    
    parser.add_argument("--seed", type=int, default=42, help="")
    parser.add_argument("--cuda-device", type=int, default=0, help="")
    
    parser.add_argument("--batch-size", type=int, default=256, help="")
    parser.add_argument("--num-epochs", type=int, default=1000, help="")
    parser.add_argument("--lr", type=float, default=0.1, help="")
    parser.add_argument("--momentum", type=float, default=0.9, help="")
    parser.add_argument("--weight-decay", type=float, default=0.0001, help="")
    parser.add_argument("--eval-step", type=int, default=1, help="")
    
    parser.add_argument("--model-dir", type=str, default="/data/users/yn621/models/ISO")
    
    return parser.parse_args()

def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True