import torch
import torch.nn as nn
from adic_components.prototype3 import LoRA

class CAttnWithLoRA(nn.Module):
    '''
    Wraps a module with a LoRA layer, for fine-tuning pretrained models, mainly the GPT-2.
    '''
    def __init__(self, original: nn.Module, d_in: int, d_out: int, r: int = 4, alpha: float = 1.0, dropout: float = 0.0):
        super().__init__()
        self.original = original
        self.lora = LoRA(d_in, d_out, r=r, alpha=alpha, dropout=dropout)

    def forward(self, x):
        return self.original(x) + self.lora(x)