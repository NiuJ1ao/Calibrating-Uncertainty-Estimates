import glob
import os

import torch
from torchvision import models

import logger as logging
from data_loader import get_CIFAR10, get_SVHN
from logger import logger
from util import args_parser, set_all_seeds


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

dataset = "val"
data_loader = data_loaders[dataset]

device = torch.device(f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu")

model = models.resnet101(num_classes=10)
class_paths = os.path.join(args.model_dir, f"{args.model}_{args.dataset}_{args.seed}_*.pt")
for class_path in glob.iglob(class_paths):
    logger.info(f"Loading model from {class_path}")
    model.load_state_dict(torch.load(class_path, map_location=device))
    model = model.to(device)
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

    logits = logits.detach().cpu()
    references = references.detach().cpu()
    
    torch.save(logits, os.path.join(args.data_path, f"{dataset}_uncalibrated_logits_{args.seed}.pt"))
    torch.save(references, os.path.join(args.data_path, f"{dataset}_labels_{args.seed}.pt"))
    