import glob
import os

import torch

import logger as logging
from calibration_metrics import expected_calibration_error, accuracy, multiclass_ece
from data_loader import get_SOLAR, get_UCI_concrete, get_UCI_energy_efficiency, get_UCI_housing, get_UCI_power, get_UCI_red_wine, get_UCI_yacht
from logger import logger
from regressor import MLP
from util import args_parser, set_all_seeds

def predict(model, data_loader, criterion, device):
    logger.info(f"{device} is available")
    
    model = model.to(device)

    with torch.no_grad():
        inputs = data_loader[0].to(device)
        labels = data_loader[1].to(device)

        logits = model(inputs)
        loss = criterion(logits, labels)
    
    logger.info('Test loss: {:4f}'.format(loss))
    
    return logits.detach()
    
def main():
    logging.init_logger(log_level=logging.DEBUG)
    args = args_parser()
    logger.info(args)
    set_all_seeds(args.seed)
    
    device = torch.device(f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu")
    
    if args.dataset == "house":
        data_loaders = get_UCI_housing(args.data_path)
    elif args.dataset == "concrete":
        data_loaders = get_UCI_concrete(args.data_path)
    elif args.dataset == "energy":
        data_loaders = get_UCI_energy_efficiency(args.data_path)
    elif args.dataset == "power":
        data_loaders = get_UCI_power(args.data_path)
    elif args.dataset == "wine":
        data_loaders = get_UCI_red_wine(args.data_path)
    elif args.dataset == "yacht":
        data_loaders = get_UCI_yacht(args.data_path)
    elif args.dataset == "solar":
        data_loaders = get_SOLAR(args.data_path)
    else:
        raise FileNotFoundError
    data_loader = data_loaders["test"]
    
    in_dim = data_loader[0].shape[1]
    model = MLP(input_dim=in_dim, output_dim=1, num_units=args.hidden_size, drop_prob=args.dropout)

    paths = os.path.join(args.model_dir, f"{args.model}_{args.dataset}_*.pt")
    preds = []
    for path in glob.iglob(paths):
        logger.info(f"Loading model from {path}")
        model.load_state_dict(torch.load(path, map_location=device))
        logger.info(model)
        
        criterion = torch.nn.L1Loss()
        logits = predict(model, data_loader, criterion, device)
        
        preds.append(logits.unsqueeze(0))
        
    preds = torch.cat(preds, dim=0)


if __name__ == "__main__":
    main()
    