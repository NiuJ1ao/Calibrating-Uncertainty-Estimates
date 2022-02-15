from logger import logger
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import transforms

transform = transforms.Compose([transforms.ToTensor()])

def get_CIFAR10(path, batch_size):
    train_CIFAR = datasets.CIFAR10(path, train=True, transform=transform)
    test_CIFAR = datasets.CIFAR10(path, train=False, transform=transform)
    train_data_loader = initialise_data_loader(train_CIFAR, batch_size)
    test_data_loader = initialise_data_loader(test_CIFAR, batch_size)
    
    logger.info(f"CIFAR-10: {len(train_data_loader.dataset)} train data, {len(test_data_loader.dataset)} test data are loaded")
    return {"train": train_data_loader, "test": test_data_loader}

def get_SVHN(path, batch_size):
    train_SVHN = datasets.SVHN(path, "train", transform=transform)
    test_SVHN = datasets.SVHN(path, "test", transform=transform)
    extra_SVHN = datasets.SVHN(path, "extra", transform=transform)
    
    train_data_loader = initialise_data_loader(train_SVHN, batch_size)
    test_data_loader = initialise_data_loader(test_SVHN, batch_size)
    extra_data_loader = initialise_data_loader(extra_SVHN, batch_size)
    
    logger.info(f"SVHN: {len(train_data_loader.dataset)} train data, {len(test_data_loader.dataset)} test data, {len(extra_data_loader.dataset)} extra data are loaded")
    return {"train": train_data_loader, "test": test_data_loader, "extra": extra_data_loader}

def initialise_data_loader(data, batch_size, n_threads=4, shuffle=True) -> DataLoader:
    return DataLoader(data, batch_size=batch_size, shuffle=shuffle, num_workers=n_threads)
    