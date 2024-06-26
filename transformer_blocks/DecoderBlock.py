import torch
import torch.nn as nn
from .MultiHeadAttention import MHA
from .AddNorm import AddNorm
from .PositionwiseFeedForward import PositionwiseFeedForward

class Decoder(nn.Module):
    def __init__(self, d_model, num_heads):
        super(Decoder, self).__init__()
        self.attention = MHA(d_model=d_model, num_heads=num_heads)
        self.addnorm1 = AddNorm(size=d_model)
        self.masked_attention = MHA(d_model=d_model, num_heads=num_heads)
        self.mha = MHA(d_model=d_model, num_heads=num_heads)
        self.addnorm2 = AddNorm(size=d_model)
        self.ffn = PositionwiseFeedForward(d_model=d_model, d_ff=d_model*6)
        self.addnorm3 = AddNorm(size=d_model)
        pass

    def create_mask(self, size):
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
        return mask[None, None, :, :]

    def forward(self, x, encoder_output):
        target_mask = self.create_mask(x.size(1))
        output, _ = self.masked_attention(x, x, x, mask=target_mask)
        x = self.addnorm1(x, output)
        cross_attention_output, _ = self.mha(x, encoder_output, encoder_output)
        x = self.addnorm2(x, cross_attention_output)
        ffn_output = self.ffn(x)
        x = self.addnorm3(x, ffn_output)
        return x