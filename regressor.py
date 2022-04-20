import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, num_units, drop_prob):
        super(MLP, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(input_dim, num_units),
            nn.ReLU(inplace = True),
            nn.Dropout(p = drop_prob),
            
            nn.Linear(num_units, num_units),
            nn.ReLU(inplace = True),
            nn.Dropout(p = drop_prob),
        )
        
        self.mu = nn.Sequential(
            nn.Linear(num_units, num_units),
            nn.ReLU(inplace = True),
            nn.Dropout(p = drop_prob),
            nn.Linear(num_units, output_dim)
        )
        self.log_sigma2 = nn.Sequential(
            nn.Linear(num_units, num_units),
            nn.ReLU(inplace = True),
            nn.Dropout(p = drop_prob),
            nn.Linear(num_units, output_dim)
        )

    def forward(self, x):
        logits = self.layers(x)
        return self.mu(logits), self.log_sigma2(logits)
    
# class MLPSigma(MLP):
#     def forward(self, x):
#         logits = super(MLPSigma, self).forward(x)
#         return nn.functional.softplus(logits)
    
class QuantileMLP(nn.Module):
    def __init__(self, input_dim, output_dim, num_units, drop_prob, quantiles):
        super(QuantileMLP, self).__init__()
        
        n = len(quantiles)
        num_units *= n
        output_dim *= n
        
        self.layers = nn.Sequential(
            nn.Linear(input_dim, num_units),
            nn.ReLU(inplace = True),
            nn.Dropout(p = drop_prob),
            
            nn.Linear(num_units, num_units),
            nn.ReLU(inplace = True),
            nn.Dropout(p = drop_prob),
            
            nn.Linear(num_units, output_dim)
        )

    def forward(self, x):
        return self.layers(x)

class GaussianNLLLossWrapper(nn.Module):
    def __init__(self):
        super(GaussianNLLLossWrapper, self).__init__()
        self.loss = nn.GaussianNLLLoss()
    
    def forward(self, y_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mu = y_pred[0]
        sigma_squared = torch.exp(y_pred[1])
        
        return self.loss(mu, target, sigma_squared)

class QuantileLoss(nn.Module):
    def __init__(self, quantiles=[0.05, 0.5, 0.95]):
        super(QuantileLoss, self).__init__()
        self.quantiles = quantiles
    
    def forward(self, y_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # calculate quantile loss
        losses = []
        target = target.view(-1)
        for i, q in enumerate(self.quantiles):
            errors = target - y_pred[:, i]
            losses.append(torch.max((q - 1) * errors, q * errors).unsqueeze(-1))

        losses = torch.sum(torch.cat(losses, dim=-1), dim=-1)
        loss = torch.mean(losses)
        return loss