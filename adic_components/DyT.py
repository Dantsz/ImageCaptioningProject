from torch import nn
from torch.nn import functional as F
import torch

class DyT(nn.Module):
    '''
    DyT model inspired by the thorough pseudocode provided in the 'Transformers without Normalization' paper by FAIR.
    '''
    def __init__(self, C, init_alpha=0.5):
        super(DyT, self).__init__()
        self.C = C

        self.alpha = nn.Parameter(torch.ones(1) * init_alpha)
        self.gamma = nn.Parameter(torch.ones(C))
        self.beta = nn.Parameter(torch.ones(C))

    def forward(self, x):
        x = F.tanh(self.alpha * x)
        return self.gamma * x + self.beta
