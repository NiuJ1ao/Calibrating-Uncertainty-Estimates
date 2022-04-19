import os

import torch
import torch.nn

import logger as logging
from calibration_models import PlattCalibration, TemperatureCalibration
from logger import logger
from util import args_parser, set_all_seeds
from calibration_metrics import expected_calibration_error, accuracy, maximum_calibration_error

def train(args, model, device):
    model.fit("val", args.data_path, args.seed)
    
    _, predictions, confidences, references = model.predict("val", args.data_path, args.seed)
    acc = accuracy(predictions, references, device)
    ece = expected_calibration_error(predictions, confidences, references, device)
    mce = maximum_calibration_error(predictions, confidences, references, device)
    logger.info(f"Val: ACC = {acc:.4f}, ECE = {ece:.4f}, MCE = {mce:.4f}")
    
    _, predictions, confidences, references = model.predict("test", args.data_path, args.seed)
    acc = accuracy(predictions, references, device)
    ece = expected_calibration_error(predictions, confidences, references, device)
    mce = maximum_calibration_error(predictions, confidences, references, device)
    logger.info(f"Test: ACC = {acc:.4f}, ECE = {ece:.4f}, MCE = {mce:.4f}")
    
    output_dir = os.path.join(args.model_dir, f"{args.model}-{args.calibrate}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if args.ensemble:
        path = os.path.join(output_dir, f"{args.model}-{args.calibrate}_{args.dataset}_ensembled_{acc:.4f}_{ece:.4f}_{mce:.4f}.pt")
    else:
        path = os.path.join(output_dir, f"{args.model}-{args.calibrate}_{args.dataset}_{args.seed}_{acc:.4f}_{ece:.4f}_{mce:.4f}.pt")
    model.dump_model(path)

def main():
    logging.init_logger(log_level=logging.DEBUG)
    args = args_parser()
    logger.info(args)
    
    set_all_seeds(args.seed)

    device = torch.device(f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu")
    logger.info(f"{device} is available")
    
    if args.calibrate == "platt":
        model = PlattCalibration(is_ensembled=args.ensemble, device=device)
    elif args.calibrate == "temp":
        model = TemperatureCalibration(is_ensembled=args.ensemble, device=device)
    else:
        raise NotImplementedError
    
    train(args, model, device)
    
if __name__ == "__main__":
    main()
