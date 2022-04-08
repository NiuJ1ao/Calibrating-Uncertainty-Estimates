import os
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import numpy as np
from logger import logger
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch.utils.data import random_split

from util import basis_function, set_all_seeds

transform = transforms.Compose([transforms.ToTensor()])

def get_CIFAR10(path, batch_size):
    train_ds = datasets.CIFAR10(path, train=True, transform=transform, download=True)
    test_ds = datasets.CIFAR10(path, train=False, transform=transform, download=True)
    
    train_ds, val_ds = random_split(train_ds, [40000, 10000])
    
    train_data_loader = initialise_data_loader(train_ds, batch_size, shuffle=True)
    val_data_loader = initialise_data_loader(val_ds, batch_size)
    test_data_loader = initialise_data_loader(test_ds, batch_size)
    
    logger.info(f"CIFAR-10: {len(train_data_loader.dataset)} train data, {len(val_data_loader.dataset)} val data, {len(test_data_loader.dataset)} test data are loaded")
    return {"train": train_data_loader, "val": val_data_loader, "test": test_data_loader}

def get_SVHN(path, batch_size):
    train_ds = datasets.SVHN(path, "train", transform=transform, download=True)
    test_ds = datasets.SVHN(path, "test", transform=transform, download=True)
    extra_ds = datasets.SVHN(path, "extra", transform=transform, download=True)
    
    val_ds, _ = random_split(extra_ds, [20000, len(extra_ds)-20000])
    
    train_data_loader = initialise_data_loader(train_ds, batch_size, shuffle=True)
    val_data_loader = initialise_data_loader(val_ds, batch_size)
    test_data_loader = initialise_data_loader(test_ds, batch_size)
    
    # print(f"SVHN: {len(train_data_loader.dataset)} train data, {len(val_data_loader.dataset)} val data, {len(test_data_loader.dataset)} test data are loaded")
    logger.info(f"SVHN: {len(train_data_loader.dataset)} train data, {len(val_data_loader.dataset)} val data, {len(test_data_loader.dataset)} test data are loaded")
    return {"train": train_data_loader, "val": val_data_loader, "test": test_data_loader}

def get_SOLAR(path):
    train_path = os.path.join(path, "solar_data_train.npy")
    train_data = np.load(train_path) # (261, 2)
    val_path = os.path.join(path, "solar_data_val.npy")
    val_data = np.load(val_path) # (30, 2)
    test_path = os.path.join(path, "solar_data_test.npy")
    test_data = np.load(test_path) # (100, 2)
    
    train_X, train_y = train_data[:, 0], train_data[:, 1]
    val_X, val_y = val_data[:, 0], val_data[:, 1]
    test_X, test_y = test_data[:, 0], test_data[:, 1]
        
    train_X, train_y = numpy2torch(train_X, train_y)
    val_X, val_y = numpy2torch(val_X, val_y)
    test_X, test_y = numpy2torch(test_X, test_y)
    
    print(train_X.shape, train_y.shape)
    print(val_X.shape, val_y.shape)
    print(test_X.shape, test_y.shape)
    logger.info(f"Solar: {len(train_y)} train data, {len(val_y)} val data, {len(test_y)} test data are loaded")
    return {"train": (train_X, train_y), "val": (val_X, val_y), "test": (test_X, test_y)}

def get_UCI(path):
    train_path = os.path.join(path, "train.npy")
    train_data = np.load(train_path)
    val_path = os.path.join(path, "val.npy")
    val_data = np.load(val_path)
    test_path = os.path.join(path, "test.npy")
    test_data = np.load(test_path)
    
    train_X, train_y = train_data[:, :-1], train_data[:, -1]
    val_X, val_y = val_data[:, :-1], val_data[:, -1]
    test_X, test_y = test_data[:, :-1], test_data[:, -1]
        
    train_X, train_y = numpy2torch(train_X, train_y)
    val_X, val_y = numpy2torch(val_X, val_y)
    test_X, test_y = numpy2torch(test_X, test_y)
    
    print(train_X.shape, train_y.shape)
    print(val_X.shape, val_y.shape)
    print(test_X.shape, test_y.shape)
    logger.info(f"Solar: {len(train_y)} train data, {len(val_y)} val data, {len(test_y)} test data are loaded")
    return {"train": (train_X, train_y), "val": (val_X, val_y), "test": (test_X, test_y)}

def get_UCI_housing(path):
    path = os.path.join(path, 'housing.data')
    data = pd.read_csv(path, header=0, delimiter="\s+").values
    data = data[np.random.permutation(np.arange(len(data)))]
    return train_val_test_split(data)

def get_UCI_concrete(path):
    path = os.path.join(path, 'Concrete_Data.xls')
    data = pd.read_excel(path, header=0).values
    data = data[np.random.permutation(np.arange(len(data)))]
    return train_val_test_split(data)

def get_UCI_energy_efficiency(path):
    path = os.path.join(path, 'ENB2012_data.xlsx')
    data = pd.read_excel(path, header=0).values
    data = data[np.random.permutation(np.arange(len(data)))]
    return train_val_test_split(data)

def get_UCI_power(path):
    path = os.path.join(path, 'CCPP', 'Folds5x2_pp.xlsx')
    data = pd.read_excel(path, header=0).values
    np.random.shuffle(data)
    return train_val_test_split(data)

def get_UCI_red_wine(path):
    path = os.path.join(path, 'winequality-red.csv')
    data = pd.read_csv(path, header=1, delimiter=';').values
    data = data[np.random.permutation(np.arange(len(data)))]
    return train_val_test_split(data)

def get_UCI_yacht(path):
    path = os.path.join(path, "yacht_hydrodynamics.data")
    data = pd.read_csv(path, header=1, delimiter='\s+').values
    data = data[np.random.permutation(np.arange(len(data)))]
    return train_val_test_split(data)

def train_val_test_split(data):
    train_X, test_X, train_y, test_y = train_test_split(data[:,:-2], data[:,-1], test_size=0.1)
    train_X, val_X, train_y, val_y = train_test_split(train_X, train_y, test_size=0.1)
    
    train_X, train_y = normalise_data(train_X, train_y)
    val_X, val_y = normalise_data(val_X, val_y)
    test_X, test_y = normalise_data(test_X, test_y)
    
    train_X, train_y = numpy2torch(train_X, train_y)
    val_X, val_y = numpy2torch(val_X, val_y)
    test_X, test_y = numpy2torch(test_X, test_y)
    
    print(train_X.shape, train_y.shape)
    print(val_X.shape, val_y.shape)
    print(test_X.shape, test_y.shape)
    logger.info(f"{len(train_y)} train data, {len(val_y)} val data, {len(test_y)} test data are loaded")
    return {"train": (train_X, train_y), "val": (val_X, val_y), "test": (test_X, test_y)}

def normalise_data(X, y):
    x_means, x_stds = X.mean(axis = 0), X.std(axis = 0)
    y_means, y_stds = y.mean(axis = 0), y.std(axis = 0)
    return (X - x_means)/x_stds, (y - y_means)/y_stds

def numpy2torch(X, y):
    if len(X.shape) == 1:
        X = X.reshape(-1, 1)
    if len(y.shape) == 1:
        y = y.reshape(-1, 1)
    return torch.from_numpy(X).float(), torch.from_numpy(y).float()

def initialise_data_loader(data, batch_size, n_threads=0, shuffle=False) -> DataLoader:
    return DataLoader(data, batch_size=batch_size, shuffle=shuffle, num_workers=n_threads)

# class SolarDataset(Dataset):
#     def __init__(self, X, y):
#         self.inputs = torch.from_numpy(X).float()
#         # print(self.inputs[:10])
        
#         # apply basis function
#         # mus = torch.linspace(-200, 200, steps=256)
#         # self.inputs = basis_function(self.inputs, mus, 10)
#         # # self.inputs = basis_function(self.inputs)
#         # self.inputs += torch.cos(self.inputs)
        
#         self.labels = torch.from_numpy(y).float()
        
#         print(self.inputs.shape, self.labels.shape)
#         # print(self.inputs[:10])
        
#     def __len__(self):
#         return len(self.labels)

#     def __getitem__(self, idx):
#         return self.inputs[idx], self.labels[idx]
    
if __name__ == "__main__":
    set_all_seeds(0)
    get_UCI_housing("/mnt/e/data")
    get_UCI("/mnt/e/data/house")
    
    # train_X, train_y = data["train"]
    # train = np.hstack((train_X, train_y.reshape(-1,1)))
    # print(train.shape)
    # np.save("/mnt/e/data/yacht/train.npy", train)
    
    # val_X, val_y = data["val"]
    # val = np.hstack((val_X, val_y.reshape(-1,1)))
    # np.save("/mnt/e/data/yacht/val.npy", val)
    
    # test_X, test_y = data["test"]
    # test = np.hstack((test_X, test_y.reshape(-1,1)))
    # np.save("/mnt/e/data/yacht/test.npy", test)