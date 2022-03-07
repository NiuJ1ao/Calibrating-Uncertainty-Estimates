import argparse
import copy
import time
import torch
import logger as logging
from logger import logger
import torch.nn
from torch.optim import SGD
from data_loader import get_CIFAR10, get_SVHN
from torchvision import models
import random
import numpy as np

def set_all_seeds(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True

def save_model(path, model):
    torch.save(model, path)

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

    return parser.parse_args()

def train(args, model, data_loaders, criterion, optimizer):
    num_epochs = args.num_epochs
    dataset = args.dataset
    eval_step = args.eval_step
    seed = args.seed
    
    since = time.time()
    device = torch.device(f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu")
    logger.info(f"{device} is available")
    
    model = model.to(device)

    val_acc_history = []

    best_model = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_epoch = 0

    for epoch in range(num_epochs):
        logger.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
        logger.info('-' * 30)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in data_loaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(data_loaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(data_loaders[phase].dataset)

            logger.info('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_epoch = epoch
                best_model = copy.deepcopy(model.state_dict())
            if phase == 'test':
                val_acc_history.append(epoch_acc)

        logger.info(f"{val_acc_history}")

    time_elapsed = time.time() - since
    logger.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    logger.info('Best val Acc: {:4f}'.format(best_acc))

    # save best model weights
    torch.save(best_model, f"/data/users/yn621/models/ISO/resnet_{dataset}_{seed}_{best_epoch}_{best_acc}.pt")
    
    logger.info(f"{val_acc_history}")

def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

def main():
    logging.init_logger()
    args = args_parser()
    logger.info(args)
    
    set_all_seeds(args.seed)
    
    if args.dataset == "cifar-10":
        data_loaders = get_CIFAR10(args.data_path, args.batch_size)
    elif args.dataset == "SVHN":
        data_loaders = get_SVHN(args.data_path, args.batch_size)
    else:
        raise FileNotFoundError
    
    model = models.resnet101(pretrained=False, progress=True) 
    model.apply(init_weights)
    
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay) 
    loss_fn = torch.nn.CrossEntropyLoss()
    
    train(args, model, data_loaders, loss_fn, optimizer)
    
if __name__ == "__main__":
    main()
