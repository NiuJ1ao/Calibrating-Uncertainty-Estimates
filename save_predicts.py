import glob
import os

import torch
from torchvision import models
from calibration_metrics import accuracy

import logger as logging
from data_loader import get_CIFAR10, get_SVHN
from logger import logger
from util import args_parser, set_all_seeds

logging.init_logger(log_level=logging.DEBUG)
args = args_parser()
logger.info(args)
set_all_seeds(args.seed)

device = torch.device(f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu")
dataset = "val"
if args.dataset == "cifar-10":
    data_loaders = get_CIFAR10(args.data_path, args.batch_size)
elif args.dataset == "SVHN":
    data_loaders = get_SVHN(args.data_path, args.batch_size)
else:
    raise FileNotFoundError
data_loader = data_loaders[dataset]
model = models.resnet101(num_classes=10)

def predict(model, class_path, data_loader, device):
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
        
    epoch_acc = running_corrects.double() / len(references)
    logger.info('Acc of uncalibrated classifier: {:.4f}'.format(epoch_acc))

    logits = logits.detach().cpu()
    references = references.detach().cpu()
    
    return logits, references

if args.ensemble: 
    all_logits = []
    class_paths = os.path.join(args.model_dir, args.model, f"{args.model}_{args.dataset}_*.pt")
    for class_path in glob.iglob(class_paths):
        logits, labels = predict(model, class_path, data_loader, device)
        all_logits.append(logits.unsqueeze(0))
    
    all_logits = torch.cat(all_logits, dim=0)
    all_logits = all_logits.mean(dim=0)
    _, preds = torch.max(all_logits, dim=-1)
    acc = accuracy(preds, labels)
    logger.info(f"Acc of ensembled classifiers: {acc:.4f}")

    print(all_logits.shape, labels.shape)
    
    torch.save(all_logits, os.path.join(args.data_path, f"{dataset}_uncalibrated_logits_ensembled.pt"))
    torch.save(labels, os.path.join(args.data_path, f"{dataset}_labels_ensembled.pt"))
    
else:
    class_paths = os.path.join(args.model_dir, args.model, f"{args.model}_{args.dataset}_{args.seed}_*.pt")
    assert len(glob.glob(class_paths)) == 1
    class_path = glob.glob(class_paths)[0]
    logits, labels = predict(model, class_path, data_loader, device)
    torch.save(logits, os.path.join(args.data_path, f"{dataset}_uncalibrated_logits_{args.seed}.pt"))
    torch.save(labels, os.path.join(args.data_path, f"{dataset}_labels_{args.seed}.pt"))
    