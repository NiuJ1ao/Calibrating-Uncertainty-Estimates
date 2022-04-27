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

class GaussianNLLLossWrapper(nn.Module):
    def __init__(self):
        super(GaussianNLLLossWrapper, self).__init__()
        self.loss = nn.GaussianNLLLoss()
    
    def forward(self, y_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mu = y_pred[0]
        sigma_squared = torch.exp(y_pred[1])
        
        return self.loss(mu, target, sigma_squared)
    
class QuantileMLP(nn.Module):
    def __init__(self, input_dim, output_dim, num_units, drop_prob):
        super(QuantileMLP, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(input_dim, num_units),
            nn.ReLU(inplace = True),
            nn.Dropout(p = drop_prob),
            
            nn.Linear(num_units, num_units),
            nn.ReLU(inplace = True),
            nn.Dropout(p = drop_prob),
            
            nn.Linear(num_units, num_units),
            nn.ReLU(inplace = True),
            nn.Dropout(p = drop_prob),
            
            nn.Linear(num_units, num_units),
            nn.ReLU(inplace = True),
            nn.Dropout(p = drop_prob),
            
            nn.Linear(num_units, output_dim * 3)
        )
        
        # self.layers = nn.Sequential(
        #     nn.Linear(input_dim, num_units),
        #     nn.ReLU(inplace = True),
        #     nn.Dropout(p = drop_prob),
            
        #     nn.Linear(num_units, num_units),
        #     nn.ReLU(inplace = True),
        #     nn.Dropout(p = drop_prob),
        # )
        
        # self.upper = nn.Sequential(
        #     nn.Linear(num_units, num_units),
        #     nn.ReLU(inplace = True),
        #     nn.Dropout(p = drop_prob),
        #     nn.Linear(num_units, output_dim)
        # )
        
        # self.median = nn.Sequential(
        #     nn.Linear(num_units, num_units),
        #     nn.ReLU(inplace = True),
        #     nn.Dropout(p = drop_prob),
        #     nn.Linear(num_units, output_dim)
        # )
        
        # self.lower = nn.Sequential(
        #     nn.Linear(num_units, num_units),
        #     nn.ReLU(inplace = True),
        #     nn.Dropout(p = drop_prob),
        #     nn.Linear(num_units, output_dim)
        # )

    def forward(self, x):
        logits = self.layers(x)
        return logits
        # return self.lower(logits), self.median(logits), self.upper(logits)

class QuantileLoss(nn.Module):
    def __init__(self, alpha):
        super(QuantileLoss, self).__init__()
        self.quantiles = [alpha/2, 0.5, 1 - alpha/2]
    
    def forward(self, y_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        losses = []
        target = target.view(-1)
        for i, q in enumerate(self.quantiles):
            errors = target - y_pred[:, i]
            losses.append(torch.max((q - 1) * errors, q * errors).unsqueeze(-1))
        # for i, q in enumerate(self.quantiles):
        #     errors = target - y_pred[i]
        #     losses.append(torch.max((q - 1) * errors, q * errors).unsqueeze(-1))

        losses = torch.sum(torch.cat(losses, dim=-1), dim=-1)
        loss = torch.mean(losses)
        return loss