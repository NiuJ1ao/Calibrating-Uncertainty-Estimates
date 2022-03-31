import torch
import pickle
from torch import nn
from logger import logger
from sklearn.linear_model import LogisticRegression

class AbstractCalibration(nn.Module):
    def __init__(self, classifier, device):
        super(AbstractCalibration, self).__init__()
        self.device = device
        
        self.classifier = classifier.to(device)
        self.classifier.eval()
        
        self.softmax = nn.Softmax(dim=-1)

    def classifier_predict(self, dataloader):
        self.classifier.eval()
        running_corrects = 0
        
        logits = []
        references = []
        
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                references.append(labels)

                b_logits = self.classifier(inputs)
                logits.append(b_logits)
                
                _, preds = torch.max(b_logits, dim=-1)

                running_corrects += torch.sum(preds == labels.data)
            logits = torch.cat(logits, dim=0) # num_samples x num_classes
            references = torch.cat(references, dim=0) # num_samples x 1
            
        epoch_acc = running_corrects.double() / len(dataloader.dataset)
        logger.info('Acc of uncalibrated classifier: {:.4f}'.format(epoch_acc))
        
        return logits.detach(), references.detach()

    def fit(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError
        
    def dump_model(self):
        raise NotImplementedError
    
    
class TemperatureCalibration(AbstractCalibration):
    def __init__(self, classifier, device, calibrator=None):
        super().__init__(classifier, device)
        if calibrator == None:
            self.temperature = torch.rand(1, requires_grad=True, device=device)
        else:
            self.temperature = calibrator
        self.relu = nn.ReLU()
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=1000)
                
    def temperature_scale(self, logits):
        temp = self.relu(self.temperature)
        temp = temp + torch.tensor(1e-5, requires_grad=False, device=self.device)
        return self.softmax(logits / temp)
    
    def fit(self, dataloader):
        criterion = self.criterion
        optimizer = self.optimizer
        
        logits, labels = self.classifier_predict(dataloader)
        
        # optimizer.zero_grad()
        # calibrated_logits = self.temperature_scale(logits)
        # loss = criterion(calibrated_logits, labels)
        # loss.backward()
        # optimizer.step()
        
        def _fit_temp():
            optimizer.zero_grad()
            loss = criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss
        optimizer.step(_fit_temp)
        
        logger.debug(f"Temperature: {self.temperature}")
        
        # for epoch in range(num_epochs):
            # logger.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
            # logger.info('-' * 30)
            # optimizer.zero_grad()
            # calibrated_logits = self.temperature_scale(logits)
            # loss = criterion(calibrated_logits, labels)
            # loss.backward()
            # optimizer.step()
            # logger.info('Loss: {:.4f}'.format(loss))
            
    def predict(self, dataloader):
        logits, references = self.classifier_predict(dataloader)
        with torch.no_grad():
            calibrated_logits = self.temperature_scale(logits)
        probabilities = self.softmax(calibrated_logits)
        confidences, predictions = torch.max(probabilities, dim=-1)
        
        return probabilities.cpu(), predictions.cpu(), confidences.cpu(), references.cpu()
        
    def dump_model(self, path):
        torch.save(self.temperature, path+".pt")
        

class PlattCalibration(AbstractCalibration):
    def __init__(self, classifier, device, calibrator=None):
        super().__init__(classifier, device)
        if calibrator == None:
            self.calibrator = LogisticRegression(penalty="none", solver='lbfgs', multi_class='multinomial')
        else:
            self.calibrator = calibrator
        
    def fit(self, dataloader):
        logits, references = self.classifier_predict(dataloader)
        logits = logits.cpu().numpy()
        references = references.cpu().numpy()
        self.calibrator.fit(logits, references)
        
    def predict(self, dataloader):
        logits, references = self.classifier_predict(dataloader)
        logits = logits.cpu().numpy()
        references = references.cpu().numpy()
        
        probs = self.calibrator.predict_proba(logits)
        probabilities = torch.from_numpy(probs)
        confidences, predictions = torch.max(probabilities, dim=-1)
        
        return probabilities, predictions, confidences, torch.from_numpy(references)
        
    def dump_model(self, path):
        pickle.dump(self.calibrator, open(path+".pkl", 'wb'))
        