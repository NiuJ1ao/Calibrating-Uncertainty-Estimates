from torch import nn

class PlattCalibration(nn.Module):
    def __init__(self, classifier):
        super(PlattCalibration, self).__init__()
        # self.classifier = classifier
        self.classifier = self.dfs_freeze(classifier)
        self.affine_layer = nn.Linear(in_features=1000, out_features=10)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        logits = self.classifier(x)
        logits = self.affine_layer(logits)
        probs = self.softmax(logits)
        return probs
        
    def dfs_freeze(self, model):
        for param in model.parameters():
            param.requires_grad = False
        return model
    
    def loss_function(self):
        return nn.NLLLoss()