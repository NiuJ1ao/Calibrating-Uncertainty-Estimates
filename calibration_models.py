import os
import torch
import pickle
from torch import nn
from calibration_metrics import expected_calibration_error
from logger import logger
from sklearn.linear_model import LogisticRegression

class AbstractCalibration(nn.Module):
    def __init__(self, device="cpu"):
        super(AbstractCalibration, self).__init__()
        self.device = device
        self.calibrator = None
        self.best_model = None
    
    def load_uncalibrated_logits(self, dataset, data_dir, seed):
        path = os.path.join(data_dir, f"{dataset}_uncalibrated_logits_{seed}.pt")
        logger.info(f"Loading uncalibrated logits from {path}")
        return torch.load(path, map_location=self.device)
        
    def load_labels(self, dataset, data_dir, seed):
        path = os.path.join(data_dir, f"{dataset}_labels_{seed}.pt")
        return torch.load(path, map_location=self.device)
    
    def train_val_split(self, logits, labels):
        data_size = len(labels)
        val_size = int(len(logits)*0.1)
        permutation = torch.randperm(data_size, device=self.device)
        train_indices = permutation[:val_size]
        val_indices = permutation[val_size:]
        logger.debug(f"{train_indices.shape}, {val_indices.shape}")
        
        train_logits = torch.index_select(logits, dim=0, index=train_indices)
        train_labels = torch.index_select(labels, dim=0, index=train_indices)
        val_logits = torch.index_select(logits, dim=0, index=val_indices)
        val_labels = torch.index_select(labels, dim=0, index=val_indices)

        return train_logits, train_labels, val_logits, val_labels
    
    def fit(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError
        
    def dump_model(self):
        raise NotImplementedError
    
    
class TemperatureCalibration(AbstractCalibration):
    def __init__(self, calibrator=None, device="cpu"):
        super().__init__(device)
        if calibrator == None:
            # self.temperature = torch.rand(1, requires_grad=True, device=device)
            self.temperature = torch.tensor(1.5, requires_grad=True, device=device)
            logger.debug(f"initial temperature: {self.temperature}")
        else:
            self.temperature = torch.load(calibrator, map_location=device)
        self.activation = nn.Softplus()
        self.softmax = nn.Softmax(dim=-1)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam([self.temperature], lr=0.01)
        
    def temperature_scale(self, logits):
        temp = self.activation(self.temperature)
        temp = temp + torch.tensor(1e-9, requires_grad=False, device=self.device)
        return self.softmax(logits / temp)
    
    def fit(self, dataset, data_dir, seed):
        logits = self.load_uncalibrated_logits(dataset, data_dir, seed)
        labels = self.load_labels(dataset, data_dir, seed)
        
        train_logits, train_labels, val_logits, val_labels = self.train_val_split(logits, labels)
        
        best_ece = 1
        best_acc = 0
        for epoch in range(200):
            self.optimizer.zero_grad()
            outputs = self.temperature_scale(train_logits)
            loss = self.criterion(outputs, train_labels)
            loss.backward()
            self.optimizer.step()
            
            with torch.no_grad():
                outputs = self.temperature_scale(val_logits)
                loss = self.criterion(outputs, val_labels)
                confs, preds = torch.max(outputs, 1)
                corrects = torch.sum(preds == val_labels)
                
            epoch_loss = loss / len(labels)
            epoch_acc = corrects.double() / len(labels)
            epoch_ece = expected_calibration_error(preds, confs, val_labels, self.device)
            
            if epoch % 10 == 0:
                logger.info('Epoch {} Val - Loss: {} Acc: {:.4f} ECE: {:.4f}'.format(epoch, epoch_loss, epoch_acc, epoch_ece))
                
            if epoch_ece < best_ece:
                best_ece = epoch_ece
                best_acc = epoch_acc
                self.best_model = self.temperature.detach()
        
        logger.info('Best val acc: {:4f}, Best val ece: {:4f}'.format(best_acc, best_ece))
            
    def predict(self, dataset, data_dir, seed):
        logits = self.load_uncalibrated_logits(dataset, data_dir, seed)
        labels = self.load_labels(dataset, data_dir, seed)
        
        if self.best_model != None:
            self.temperature = self.best_model
        
        with torch.no_grad():
            calibrated_logits = self.temperature_scale(logits)
            probabilities = self.softmax(calibrated_logits)
            confidences, predictions = torch.max(probabilities, dim=-1)
        
        return probabilities, predictions, confidences, labels
        
    def dump_model(self, path):
        torch.save(self.temperature, path+".pt")
        

class PlattCalibration(AbstractCalibration):
    def __init__(self, calibrator=None, device="cpu"):
        super().__init__(device)
        # self.calibrator = LogisticRegression(penalty="none", solver='lbfgs', multi_class='multinomial')
        self.calibrator = nn.Sequential(
            nn.Linear(in_features=10, out_features=10, bias=True, device=self.device),
            nn.Softmax(dim=-1)
        )
        if calibrator != None:
            self.calibrator.load_state_dict(torch.load(calibrator, map_location=device))
        
        logger.info(self.calibrator)
            
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.calibrator.parameters(), lr=0.01)
        
    def fit(self, dataset, data_dir, seed):
        logits = self.load_uncalibrated_logits(dataset, data_dir, seed)
        labels = self.load_labels(dataset, data_dir, seed)
        
        train_logits, train_labels, val_logits, val_labels = self.train_val_split(logits, labels)
        
        best_ece = 1
        best_acc = 0
        for epoch in range(100):
            self.optimizer.zero_grad()
            outputs = self.calibrator(train_logits)
            loss = self.criterion(outputs, train_labels)
            loss.backward()
            self.optimizer.step()
            
            with torch.no_grad():
                outputs = self.calibrator(val_logits)
                loss = self.criterion(outputs, val_labels)
                confs, preds = torch.max(outputs, 1)
                corrects = torch.sum(preds == val_labels)
                
            epoch_loss = loss / len(labels)
            epoch_acc = corrects.double() / len(labels)
            epoch_ece = expected_calibration_error(preds, confs, val_labels, self.device)
            
            if epoch % 10 == 0:
                logger.info('Epoch {} Val - Loss: {} Acc: {:.4f} ECE: {:.4f}'.format(epoch, epoch_loss, epoch_acc, epoch_ece))
                
            if epoch_ece < best_ece:
                best_ece = epoch_ece
                best_acc = epoch_acc
                self.best_model = self.calibrator.state_dict()
        
        logger.info('Best val acc: {:4f}, Best val ece: {:4f}'.format(best_acc, best_ece))
        
    def predict(self, dataset, data_dir, seed):
        logits = self.load_uncalibrated_logits(dataset, data_dir, seed)
        labels = self.load_labels(dataset, data_dir, seed)
        
        if self.best_model != None:
            self.calibrator.load_state_dict(self.best_model)
        
        probabilities = self.calibrator(logits)
        confidences, predictions = torch.max(probabilities, dim=-1)
        
        return probabilities, predictions, confidences, labels
        
    def dump_model(self, path):
        if self.best_model != None:
            torch.save(self.best_model, path)
        else:
            torch.save(self.calibrator.state_dict(), path)
        
        