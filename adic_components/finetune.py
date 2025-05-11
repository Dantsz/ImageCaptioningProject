import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LoRA(nn.Module):
    def __init__(self, d_in: int, d_out: int, r: int = 4, alpha: float = 1.0, dropout: float = 0.0):
        super(LoRA, self).__init__()
        self.r = r
        self.alpha = alpha
        self.scale = alpha / r
        self.lora_A = nn.Linear(d_in, r, bias=False)
        self.lora_B = nn.Linear(r, d_out, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        lora_out = self.lora_B(self.lora_A(self.dropout(x))) * self.scale
        return lora_out

class LoRAdLMHead(nn.Module):
    def __init__(self, lm_head: nn.Embedding, r: int = 4, alpha: float = 1.0, dropout: float = 0.0):
        super(LoRAdLMHead, self).__init__()
        self.lm_head = lm_head
        self.r = r
        self.alpha = alpha
        self.scale = alpha / r
        d, vocab = lm_head.embedding_dim, lm_head.num_embeddings
        self.lora = LoRA(d, vocab, r=r, alpha=alpha, dropout=dropout)
        self.lm_head.weight.requires_grad = False

    def forward(self, x):
        base_out = F.linear(x, self.lm_head.weight)  # x: [B, T, D]
        lora_out = self.lora(x)  # lora_out: [B, T, V]
        return base_out + lora_out
class CAttnWithLoRA(nn.Module):
    '''
    Wraps a module with a LoRA layer, for fine-tuning pretrained models, mainly the GPT-2.
    '''
    def __init__(self, original: nn.Module, d_in: int, d_out: int, r: int = 4, alpha: float = 1.0, dropout: float = 0.0):
        super(CAttnWithLoRA, self).__init__()
        self.original = original
        self.lora = LoRA(d_in, d_out, r=r, alpha=alpha, dropout=dropout)

    def forward(self, x):
        return self.original(x) + self.lora(x)
