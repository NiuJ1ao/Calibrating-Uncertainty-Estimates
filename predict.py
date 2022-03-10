import glob
import os

import torch
from torchvision import models
from calibration_models import PlattCalibration

import logger as logging
from calibration_metrics import expected_calibration_error
from data_loader import get_CIFAR10, get_SVHN
from logger import logger
from util import args_parser, set_all_seeds

def load_model(path, device):
    logger.info(f"Loading model from {path}")
    # model = models.resnet101()
    # model.load_state_dict(torch.load(path, map_location=device))
    model = PlattCalibration(models.resnet101())
    model.load_state_dict(torch.load(path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def predict(model, data_loader, device):
    probabilities, references = [], []
    for inputs, labels in data_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            probs = outputs.softmax(dim=-1) # b x c
            probabilities.append(probs)
            
        references.append(labels)
    
    probabilities = torch.vstack(probabilities)
    references = torch.cat(references)
    logger.debug(probabilities.shape) # n x c
    logger.debug(references.shape) # n x c
    return probabilities, references

def main():
    logging.init_logger(log_level=logging.DEBUG)
    args = args_parser()
    logger.info(args)
    set_all_seeds(args.seed)
    
    if args.dataset == "cifar-10":
        data_loaders = get_CIFAR10(args.data_path, args.batch_size)
    elif args.dataset == "SVHN":
        data_loaders = get_SVHN(args.data_path, args.batch_size)
    else:
        raise FileNotFoundError
    data_loader = data_loaders["test"]
    
    probabilities = []
    path = os.path.join(args.model_dir, f"resnet-platt_{args.dataset}_*.pt")
    for filepath in glob.iglob(path):
        device = torch.device(f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu")
        model = load_model(filepath, device)
        probs, references = predict(model, data_loader, device)
        probabilities.append(probs.unsqueeze(0))
    
    probabilities = torch.vstack(probabilities)
    logger.debug(probabilities.shape)

    probs_mean = probabilities.mean(dim=0)
    probs_std = probabilities.std(dim=0)
    
    confidences, predictions = torch.max(probs_mean, dim=1)
     
    predictions = predictions.cpu()
    confidences = confidences.cpu()
    references = references.cpu()
    
    # ECE
    logger.debug(f"{predictions.shape}, {confidences.shape}, {references.shape}")
    ece = expected_calibration_error(predictions=predictions, references=references, confidences=confidences)
    logger.info(f"ECE = {ece}")
    
    output_dir = f"outputs/resnet-platt_{args.dataset}_{ece:.4f}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    torch.save(predictions, f"{output_dir}/predictions.pt")
    torch.save(confidences, f"{output_dir}/confidences.pt")
    torch.save(references, f"{output_dir}/references.pt")
        
if __name__ == "__main__":
    main()
    