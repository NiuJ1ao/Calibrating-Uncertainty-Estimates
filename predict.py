import glob
import os
import pickle

import torch
from torchvision import models
from calibration_models import PlattCalibration, TemperatureCalibration

import logger as logging
from calibration_metrics import expected_calibration_error, accuracy, multiclass_ece
from data_loader import get_CIFAR10, get_SVHN
from logger import logger
from util import args_parser, save_outputs, set_all_seeds

def load_model(path, device, calibration=None):
    logger.info(f"Loading model from {path}")
    if calibration == "platt":
        model = PlattCalibration(calibrator=path, device=device)
    elif calibration == "temp":
        model = TemperatureCalibration(calibrator=path, device=device)
    else:
        model = models.resnet101(num_classes=10)
        model.load_state_dict(torch.load(path, map_location=device))
        model = model.to(device)
        model.eval()

    return model

def uncali_predict(model, data_loader, device):
    model.eval()
    running_corrects = 0
        
    logits = []
    references = []
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            references.append(labels)

            b_logits = model(inputs)
            logits.append(b_logits)
            
            _, preds = torch.max(b_logits, dim=-1)

            running_corrects += torch.sum(preds == labels.data)
        logits = torch.cat(logits, dim=0) # num_samples x num_classes
        references = torch.cat(references, dim=0) # num_samples x 1
        
    epoch_acc = running_corrects.double() / len(data_loader.dataset)
    logger.info('Acc of uncalibrated classifier: {:.4f}'.format(epoch_acc))
    
    return logits.detach(), references.detach()

def main():
    logging.init_logger(log_level=logging.DEBUG)
    args = args_parser()
    logger.info(args)
    set_all_seeds(args.seed)

    device = torch.device(f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu")
    calibration = args.calibrate
    
    if calibration == None: 
        probabilities = []
        model_name = f"{args.model}"
        paths = os.path.join(args.model_dir, f"{model_name}_{args.dataset}_*.pt")
        for path in glob.iglob(paths):
            model = load_model(path, device, calibration)
            
            if args.dataset == "cifar-10":
                data_loaders = get_CIFAR10(args.data_path, args.batch_size)
            elif args.dataset == "SVHN":
                data_loaders = get_SVHN(args.data_path, args.batch_size)
            else:
                raise FileNotFoundError
            data_loader = data_loaders["test"]
            
            probs, references = uncali_predict(model, data_loader, device)
            probs = probs.softmax(dim=-1)
            probabilities.append(probs.unsqueeze(0))
    else: 
        model_name = f"{args.model}-{calibration}"
        paths = os.path.join(args.model_dir, f"{model_name}_{args.dataset}_*.pt")
        probabilities = []
        for path in glob.iglob(paths):
            seed = path.split("_")[-3]
            model = load_model(path, device, calibration)
            probs, _, _, references = model.predict("test", args.data_path, seed)
            probabilities.append(probs.unsqueeze(0))
    
    probabilities = torch.vstack(probabilities)
    logger.debug(probabilities.shape)

    # take mean of all models
    probs_mean = probabilities.mean(dim=0)
    probs_std = probabilities.std(dim=0)
    
    confidences, predictions = torch.max(probs_mean, dim=1)
     
    predictions = predictions.cpu()
    confidences = confidences.cpu()
    references = references.cpu()
    
    # ECE
    ece = expected_calibration_error(predictions=predictions, references=references, confidences=confidences)
    logger.info(f"ECE = {ece}")
    
    acc = accuracy(predictions=predictions, references=references)
    logger.info(f"ACC = {acc}")
    
    output_dir = f"outputs/{model_name}_{args.dataset}_{ece:.4f}"
    save_outputs(output_dir, predictions, confidences, references)

if __name__ == "__main__":
    main()
    