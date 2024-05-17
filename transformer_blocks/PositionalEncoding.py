import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=50):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len


        self.pos_encoding = torch.zeros(max_len, d_model)
        positions = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        self.pos_encoding[:, 0::2] = torch.sin(positions * div_term)
        self.pos_encoding[:, 1::2] = torch.cos(positions * div_term)

        self.register_buffer('penc', self.pos_encoding)       

    def forward(self, x):
        
        seq_len = x.size(1)
        pos_enc = self.pos_encoding[:seq_len, :].unsqueeze(0).to(x.device)
        # pos_enc = self.pos_encoding[:seq_len, :].unsqueeze(0)
        return x + pos_enc

    


