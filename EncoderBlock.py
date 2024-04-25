import torch.nn as nn
from PositionalEncoding import PositionalEncoding
from MultiHeadAttention import MHA
from AddNorm import AddNorm
from PositionwiseFeedForward import PositionwiseFeedForward

class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.pos_enc = PositionalEncoding(d_model=embedding_dim, max_len=50)
        self.mha = MHA(d_model=embedding_dim, n_heads=5)
        self.addnorm = AddNorm(size=embedding_dim)
        self.ffn = PositionwiseFeedForward(d_model=embedding_dim, d_ff=embedding_dim*6)
        self.addnorm2 = AddNorm(size=embedding_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_enc(x)
        mha_output, attn_weights = self.mha(x)
        x = self.addnorm(x, mha_output)
        x = self.ffn(x)
        x = self.addnorm2(x, x)
        return x
        
        