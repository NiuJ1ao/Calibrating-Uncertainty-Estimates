import os
import argparse
import random
import numpy as np
import torch

UCI = ["house", "concrete", "energy", "power", "red_wine", "yacht"]

def args_parser():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--data-path", type=str, default="/data/users/yn621/cifar-10", help="The root path to data directory")
    parser.add_argument("--dataset", type=str, default="cifar-10", 
                        choices=["cifar-10", "SVHN", "solar"] + UCI, 
                        help="")
    parser.add_argument("--model-dir", type=str, default="/data/users/yn621/models/ISO")
    parser.add_argument("--model", type=str, default="resnet", help="")
    parser.add_argument("--ensemble", action="store_true", help="")
    parser.add_argument("--calibrate", type=str, default=None, choices=[None, "platt", "temp"], help="")
    
    parser.add_argument("--seed", type=int, default=42, help="")
    parser.add_argument("--cuda-device", type=int, default=0, help="")
    
    parser.add_argument("--batch-size", type=int, default=256, help="")
    parser.add_argument("--num-epochs", type=int, default=1000, help="")
    parser.add_argument("--lr", type=float, default=0.1, help="")
    parser.add_argument("--momentum", type=float, default=0.9, help="")
    parser.add_argument("--weight-decay", type=float, default=0.0001, help="")
    parser.add_argument("--dropout", type=float, default=0.1, help="")
    parser.add_argument("--hidden-size", type=int, default=10, help="")
    parser.add_argument("--eval-step", type=int, default=1, help="")
    parser.add_argument("--early-stop", type=int, default=10, help="")
    
    return parser.parse_args()

def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
def save_outputs(output_dir, predictions, confidences, references):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    predictions = predictions.cpu()
    confidences = confidences.cpu()
    references = references.cpu()
        
    torch.save(predictions, f"{output_dir}/predictions.pt")
    torch.save(confidences, f"{output_dir}/confidences.pt")
    torch.save(references, f"{output_dir}/references.pt")
    
def basis_function(x, mu, l):
    return torch.exp(-torch.abs(x - mu)**2 / l)

# def basis_function(x, device):
#     x = x.expand(-1, 10)
#     pows = torch.arange(0, x.size(1), device=device)
#     return torch.pow(x, pows)