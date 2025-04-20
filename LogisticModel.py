import torch
import torch.nn.functional as F
import torch.nn as nn

class LogisticModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1) #output a single logit
        self.input_dim = input_dim


    def forward(self, x):

        x = x.view(x.size(0), -1) # Flatten the input image [batch_size, channels, height, width] -> [batch_size, input_dim]
        logits = self.linear(x) # Compute the logit using the linear layer
        return logits

