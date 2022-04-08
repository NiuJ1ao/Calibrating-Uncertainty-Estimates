import copy
import os
import glob
import pickle
import time

import torch
import torch.nn
from torch.optim import SGD, AdamW
from torchvision import models

import logger as logging
from calibration_models import PlattCalibration, TemperatureCalibration
from data_loader import get_CIFAR10, get_SVHN
from logger import logger
from util import args_parser, save_outputs, set_all_seeds
from sklearn.linear_model import LogisticRegression
from calibration_metrics import expected_calibration_error, accuracy


'''
def train(args, model, data_loaders, criterion, optimizer, device):
    num_epochs = args.num_epochs
    eval_step = args.eval_step
    
    since = time.time()
    model = model.to(device)

    val_acc_history = []

    # best_model = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    # best_epoch = 0
    early_stopping_counter = 0
    
    # train_dataloader = data_loaders["train"]
    val_dataloader = data_loaders["val"]

    # for epoch in range(num_epochs):
    #     logger.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
    #     logger.info('-' * 30)

        # model.train()
        # running_loss = 0.0
        # running_corrects = 0

        # for inputs, labels in train_dataloader:
        #     inputs = inputs.to(device)
        #     labels = labels.to(device)

        #     optimizer.zero_grad()

        #     outputs = model(inputs)
        #     loss = criterion(outputs, labels)
        #     _, preds = torch.max(outputs, 1)

        #     loss.backward()
        #     optimizer.step()

        #     running_loss += loss.item() * inputs.size(0)
        #     running_corrects += torch.sum(preds == labels.data)

        # epoch_loss = running_loss / len(train_dataloader.dataset)
        # epoch_acc = running_corrects.double() / len(train_dataloader.dataset)

        # logger.info('Train Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
        
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    
    for inputs, labels in val_dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(val_dataloader.dataset)
    epoch_acc = running_corrects.double() / len(val_dataloader.dataset)

    logger.info('Val Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

    if epoch_acc > best_acc:
        best_acc = epoch_acc
        best_epoch = epoch
        best_model = copy.deepcopy(model.state_dict())
        early_stopping_counter = 0
    else:
        early_stopping_counter += 1

    val_acc_history.append(epoch_acc.item())
    logger.info(f"{val_acc_history}")

    if early_stopping_counter >= args.early_stop:
        logger.info("Early stopping at epoch {}".format(epoch))

    time_elapsed = time.time() - since
    logger.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    logger.info('Best val acc: {:4f}'.format(best_acc))

    # save best model weights
    path = os.path.join(args.model_dir, f"{args.model}-{args.calibrate}_{args.dataset}_{args.seed}_{best_epoch}_{best_acc:.4f}.pt")
    torch.save(best_model, path)
'''

def train(args, model, device):
    model.fit("val", args.data_path, args.seed)
    
    _, predictions, confidences, references = model.predict("val", args.data_path, args.seed)
    
    ece = expected_calibration_error(predictions, confidences, references, device)
    logger.info(f"ECE = {ece:.4f}")
    
    acc = accuracy(predictions, references)
    logger.info(f"ACC = {acc:.4f}")
    
    path = os.path.join(args.model_dir, f"{args.model}-{args.calibrate}_{args.dataset}_{args.seed}_{ece:.4f}_{acc:.4f}.pt")
    model.dump_model(path)

def main():
    logging.init_logger(log_level=logging.DEBUG)
    args = args_parser()
    logger.info(args)
    
    set_all_seeds(args.seed)

    device = torch.device(f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu")
    logger.info(f"{device} is available")
    
    if args.calibrate == "platt":
        model = PlattCalibration(device=device)
    elif args.calibrate == "temp":
        model = TemperatureCalibration(device=device)
    else:
        raise NotImplementedError
    
    train(args, model, device)
    
if __name__ == "__main__":
    main()
