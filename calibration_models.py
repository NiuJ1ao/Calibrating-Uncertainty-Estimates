import os
import torch
import copy
from torch import nn
from calibration_metrics import expected_calibration_error, maximum_calibration_error
from logger import logger

class AbstractCalibration(nn.Module):
    def __init__(self, is_ensembled=False, device="cpu"):
        super(AbstractCalibration, self).__init__()
        self.device = device
        self.is_ensembled = is_ensembled
        self.best_model = None
    
    def load_uncalibrated_logits(self, dataset, data_dir, seed):
        path = os.path.join(data_dir, f"{dataset}_uncalibrated_logits_{seed}.pt")
        logger.info(f"Loading uncalibrated logits from {path}")
        return torch.load(path, map_location=self.device)
        
    def load_labels(self, dataset, data_dir, seed):
        path = os.path.join(data_dir, f"{dataset}_labels_{seed}.pt")
        return torch.load(path, map_location=self.device)
    
    def load_data(self, dataset, data_dir, seed):
        if self.is_ensembled:
            logits = self.load_uncalibrated_logits(dataset, data_dir, "ensembled")
            labels = self.load_labels(dataset, data_dir, "ensembled")
        else:
            logits = self.load_uncalibrated_logits(dataset, data_dir, seed)
            labels = self.load_labels(dataset, data_dir, seed)
        return logits, labels
    
    def train_val_split(self, logits, labels):
        data_size = len(labels)
        val_size = int(len(logits)*0.3)
        permutation = torch.randperm(data_size, device=self.device)
        val_indices = permutation[:val_size]
        train_indices = permutation[val_size:]
        
        train_logits = torch.index_select(logits, dim=0, index=train_indices)
        train_labels = torch.index_select(labels, dim=0, index=train_indices)
        val_logits = torch.index_select(logits, dim=0, index=val_indices)
        val_labels = torch.index_select(labels, dim=0, index=val_indices)
        
        logger.debug(f"{train_labels.shape}, {val_labels.shape}")

        return train_logits, train_labels, val_logits, val_labels
    
    def fit(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError
        
    def dump_model(self):
        raise NotImplementedError
    
    
class TemperatureCalibration(AbstractCalibration):
    def __init__(self, is_ensembled=False, temperature=None, device="cpu"):
        super().__init__(is_ensembled, device)
        if temperature == None:
            self.temperature = torch.tensor(1., requires_grad=True, device=device)
            # self.temperature = torch.tensor(1.5, requires_grad=True, device=device)
        else:
            self.temperature = torch.load(temperature, map_location=device)
        logger.debug(f"initial temperature: {self.temperature}")
        
        self.softmax = nn.Softmax(dim=-1)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam([self.temperature], lr=0.01)
        
    def temperature_scale(self, logits):
        return self.softmax(logits / torch.exp(self.temperature))
    
    def fit(self, dataset, data_dir, seed):
        logits, labels = self.load_data(dataset, data_dir, seed)
        train_logits, train_labels, val_logits, val_labels = self.train_val_split(logits, labels)
        
        best_ece, best_mce = 1, 1
        best_acc = 0
        for epoch in range(100):
            self.optimizer.zero_grad()
            outputs = self.temperature_scale(train_logits)
            loss = self.criterion(outputs, train_labels)
            loss.backward()
            self.optimizer.step()
            # logger.debug(f"epoch {epoch}: {self.temperature}")
            
            with torch.no_grad():
                outputs = self.temperature_scale(val_logits)
                loss = self.criterion(outputs, val_labels)
                confs, preds = torch.max(outputs, dim=-1)
                corrects = torch.sum(preds == val_labels)
                
            epoch_loss = loss / len(val_labels)
            epoch_acc = corrects.double() / len(val_labels)
            epoch_ece = expected_calibration_error(preds, confs, val_labels, self.device)
            epoch_mce = maximum_calibration_error(preds, confs, val_labels, self.device)
            
            if epoch % 10 == 0:
                logger.info('Epoch {} Val - Loss: {} Acc: {:.4f} ECE: {:.4f} MCE: {:.4f}'.format(epoch, epoch_loss, epoch_acc, epoch_ece, epoch_mce))
                
            if epoch_ece < best_ece:
                best_ece = epoch_ece
                best_mce = epoch_mce
                best_acc = epoch_acc
                logger.debug(f"better temperature: {self.temperature}")
                self.best_model = copy.deepcopy(self.temperature)
        
        logger.info('Best val acc: {:4f}, Best val ece: {:4f}, Best val mce: {:4f}'.format(best_acc, best_ece, best_mce))
            
    def predict(self, dataset, data_dir, seed):
        logits, labels = self.load_data(dataset, data_dir, seed)
        
        if self.best_model != None:
            self.temperature = self.best_model
        
        logger.debug(f"Final temperature: {self.temperature}")
        
        with torch.no_grad():
            probabilities = self.temperature_scale(logits)
            confidences, predictions = torch.max(probabilities, dim=-1)
        
        return probabilities, predictions, confidences, labels
        
    def dump_model(self, path):
        torch.save(self.temperature, path+".pt")
        

class PlattCalibration(AbstractCalibration):
    def __init__(self, is_ensembled=False, calibrator=None, device="cpu"):
        super().__init__(is_ensembled, device)
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
        logits, labels = self.load_data(dataset, data_dir, seed)    
        train_logits, train_labels, val_logits, val_labels = self.train_val_split(logits, labels)
        
        best_ece, best_mce = 1, 1
        best_acc = 0
        for epoch in range(250):
            self.optimizer.zero_grad()
            outputs = self.calibrator(train_logits)
            loss = self.criterion(outputs, train_labels)
            loss.backward()
            self.optimizer.step()
            
            with torch.no_grad():
                outputs = self.calibrator(val_logits)
                loss = self.criterion(outputs, val_labels)
                confs, preds = torch.max(outputs, dim=-1)
                corrects = torch.sum(preds == val_labels)
                
            epoch_loss = loss / len(val_labels)
            epoch_acc = corrects.double() / len(val_labels)
            epoch_ece = expected_calibration_error(preds, confs, val_labels, self.device)
            epoch_mce = maximum_calibration_error(preds, confs, val_labels, self.device)
            
            if epoch % 10 == 0:
                logger.info('Epoch {} Val - Loss: {} Acc: {:.4f} ECE: {:.4f} MCE: {:.4f}'.format(epoch, epoch_loss, epoch_acc, epoch_ece, epoch_mce))
                
            if epoch_ece < best_ece:
                best_ece = epoch_ece
                best_mce = epoch_mce
                best_acc = epoch_acc
                self.best_model = copy.deepcopy(self.calibrator.state_dict())
        
        logger.info('Best val acc: {:4f}, Best val ece: {:4f}, Best val mce: {:4f}'.format(best_acc, best_ece, best_mce))
        
    def predict(self, dataset, data_dir, seed):
        logits, labels = self.load_data(dataset, data_dir, seed)
        
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
        
        