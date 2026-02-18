import torch
import torch.nn as nn
import math

class InputEmeddings(nn.Module):

    def __init__(self, d_model:int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model:int, seq_len:int, dropout:float) -> None:
        super().__init__()

        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(seq_len, d_model)


        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model,2, dtype=torch.float)* (-math.log(10000.0)/d_model))


        pe[:,0::2] = torch.sin(position*div_term)
        pe[:,1::2] = torch.cos(position*div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer('pe',pe)

    def forward(self,x):
        x = x + (self.pe[:, :x.shape[1],:]).require_grad_(False)
        return self.dropout(x)


class layernormalization(nn.Module):

    def __init__(self, eps:float = 10**-6) -> None:
        super().__init__()
        self.eps = eps

        self.aplha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.ones(1))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)

        return self.aplha * (x-mean) / (std +self.eps) + self.bias

class feedforward(nn.Module):
    def __init__(self, d_model:int, d_ff:int, dropout:float) -> None:
        super().__init__()

        self.linear_1 = nn.Linear(d_model,d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff,d_model)

        def forward(self, x):
            return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


class MultiheadattentionBlock(nn.Module):
    def __init__(self, d_model:int, h:int, dropout:float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.dropout = dropout

        assert d_model % h == 0, "d_model is not divisible by h"

