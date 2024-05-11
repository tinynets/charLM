import torch.nn as nn
from .PositionalEncoding import PositionalEncoding
from .MultiHeadAttention import MHA
from .AddNorm import AddNorm
from .PositionwiseFeedForward import PositionwiseFeedForward

class Encoder(nn.Module):
    def __init__(self, d_model, num_heads):
        super(Encoder, self).__init__()
       
        self.pos_enc = PositionalEncoding(d_model=d_model, max_len=50)
        self.mha = MHA(d_model=d_model, num_heads=num_heads)
        self.addnorm = AddNorm(size=d_model)
        self.ffn = PositionwiseFeedForward(d_model=d_model, d_ff=d_model*6)
        self.addnorm2 = AddNorm(size=d_model)

    def forward(self, x):
        
        mha_output, attn_weights = self.mha(x, x, x)
        x = self.addnorm(x, mha_output)
        x = self.ffn(x)
        x = self.addnorm2(x, x)
        return x
    
        
        