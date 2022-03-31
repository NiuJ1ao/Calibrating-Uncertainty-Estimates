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

def load_model(class_path, cali_path, device, calibration=None):
    logger.info(f"Loading model from {class_path} and calibrator from {cali_path}")
    uncali_model = models.resnet101(num_classes=10)
    uncali_model.load_state_dict(torch.load(class_path, map_location=device))
    uncali_model = uncali_model.to(device)
    uncali_model.eval()
    
    if calibration == "platt":
        platt = pickle.load(open(cali_path, 'rb'))
        model = PlattCalibration(uncali_model, device, calibrator=platt)
    elif calibration == "temp":
        temp = torch.load(cali_path, map_location=device)
        model = TemperatureCalibration(uncali_model, device, calibrator=temp)
    else:
        return uncali_model
                    
    return model

def uncali_predict(model, data_loader, device):
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
    logger.debug(references.shape) # n x 1
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
    
    device = torch.device(f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu")
    
    calibration = args.calibrate
    class_paths = os.path.join(args.model_dir, f"{args.model}_{args.dataset}_*.pt")
    if calibration == None: 
        probabilities = []
        for class_path in glob.iglob(class_paths):
            model = load_model(class_path, device, calibration)
            probs, references = uncali_predict(model, data_loader, device)
            probabilities.append(probs.unsqueeze(0))
    else: 
        model_name = f"{args.model}-{calibration}"
        suffix = "pkl" if calibration == "platt" else "pt"  
        cali_paths = os.path.join(args.model_dir, f"{model_name}_{args.dataset}_*.{suffix}")
        probabilities = []
        for class_path, cali_path in zip(glob.iglob(class_paths), glob.iglob(cali_paths)):
            # print(class_path, cali_path)
            model = load_model(class_path, cali_path, device, calibration)
            probs, _, _, references = model.predict(data_loader)
            probabilities.append(probs.unsqueeze(0))
    
    # assert False
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
    