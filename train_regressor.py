import copy
import os
import time
import numpy as np
import torch
import torch.nn
from torch.optim import SGD, Adam

import logger as logging
from data_loader import get_SOLAR, get_UCI
from logger import logger
from regressor import MLP, QuantileLoss, QuantileMLP
from util import args_parser, set_all_seeds, UCI
from tqdm import tqdm


def save_model(path, model):
    torch.save(model, path)

def train(args, model, data_loaders, criterion, optimizer):
    num_epochs = args.num_epochs
    
    since = time.time()
    device = torch.device(f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu")
    logger.info(f"{device} is available")
    
    model = model.to(device)

    val_acc_history = []

    best_model = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    best_epoch = 0
    early_stopping = 0

    pbar = tqdm(range(num_epochs))
    for epoch in range(num_epochs):
        model.train()

        inputs = data_loaders['train'][0].to(device)
        labels = data_loaders['train'][1].to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss = loss / len(labels)

        model.eval()

        inputs = data_loaders['val'][0].to(device)
        labels = data_loaders['val'][1].to(device)

        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        val_loss = loss / len(labels)

        # deep copy the model
        if val_loss < best_loss:
            early_stopping = 0
            best_loss = val_loss
            best_epoch = epoch
            best_model = copy.deepcopy(model.state_dict())
        else:
            early_stopping += 1
            # if early_stopping >= args.early_stop:
            #     logger.info(f"Early stopping at epoch {epoch}")
            #     break
        
        pbar.update(1)
        pbar.set_postfix({"train_loss": train_loss.item(), "val_loss": val_loss.item()})
        val_acc_history.append(val_loss.item())

        logger.debug(f"{val_acc_history}")

    time_elapsed = time.time() - since
    logger.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    logger.info('Best val loss: {:4f} at epoch {}'.format(best_loss, best_epoch))
    
    model.eval()

    inputs = data_loaders['test'][0].to(device)
    labels = data_loaders['test'][1].to(device)

    with torch.no_grad():
        outputs = model(inputs)
        loss = criterion(outputs, labels)

    test_loss = loss / len(labels)
    logger.info('Test loss: {:4f}'.format(test_loss))

    # save best model weights
    path = os.path.join(args.model_dir, args.model, f"{args.model}_{args.dataset}_{args.seed}_{best_epoch}_{test_loss:.4f}.pt")
    torch.save(best_model, path)

def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)
        
def log_gaussian_loss(output, target, sigma, no_dim):
    exponent = -0.5*(target - output)**2/sigma**2
    log_coeff = -no_dim*torch.log(sigma) - 0.5*no_dim*np.log(2*np.pi)
    
    return - (log_coeff + exponent).sum()

def main():
    logging.init_logger(log_level=logging.INFO)
    args = args_parser()
    logger.info(args)
    
    set_all_seeds(args.seed)
    
    if args.dataset in UCI:
        data_loaders = get_UCI(args.data_path)
    elif args.dataset == "solar":
        data_loaders = get_SOLAR(args.data_path)
    else:
        raise FileNotFoundError
    
    in_dim = data_loaders['train'][0].shape[1]
    if args.model == "mlp":
        model = MLP(input_dim=in_dim, output_dim=1, num_units=args.hidden_size, drop_prob=args.dropout)
        loss_fn = torch.nn.L1Loss()
    elif args.model == "quantile":
        quantiles = [0.05, 0.5, 0.95]
        model = QuantileMLP(input_dim=in_dim, output_dim=1, num_units=args.hidden_size, drop_prob=args.dropout, quantiles=quantiles)
        loss_fn = QuantileLoss(quantiles)
    logger.info(model)
    
    data_len = len(data_loaders['train'][0])
    logger.debug(f"weight decay: {1e-1/data_len**0.5}")
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-1/data_len**0.5)
    train(args, model, data_loaders, loss_fn, optimizer)
    
if __name__ == "__main__":
    main()
