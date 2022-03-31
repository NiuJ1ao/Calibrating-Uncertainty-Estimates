from logger import logger
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch.utils.data import random_split

transform = transforms.Compose([transforms.ToTensor()])

# class LogitsCalibrationDataset(Dataset):
#     def __init__(self, logits, labels):
#         self.logits = logits
#         self.labels = labels

#     def __len__(self):
#         return len(self.labels)

#     def __getitem__(self, idx):
#         return self.logits[idx], self.labels[idx]

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
    
    print(f"SVHN: {len(train_data_loader.dataset)} train data, {len(val_data_loader.dataset)} val data, {len(test_data_loader.dataset)} test data are loaded")
    logger.info(f"SVHN: {len(train_data_loader.dataset)} train data, {len(val_data_loader.dataset)} val data, {len(test_data_loader.dataset)} test data are loaded")
    return {"train": train_data_loader, "val": val_data_loader, "test": test_data_loader}

def initialise_data_loader(data, batch_size, n_threads=0, shuffle=False) -> DataLoader:
    return DataLoader(data, batch_size=batch_size, shuffle=shuffle, num_workers=n_threads)
    
if __name__ == "__main__":
    get_CIFAR10("/mnt/e/data/cifar-10", 256)