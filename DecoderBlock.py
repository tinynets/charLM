import torch.nn as nn
from MultiHeadAttention import MHA
from AddNorm import AddNorm


class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.attention = nn.MHA(d_model=embedding_dim, n_heads=5)
        self.addnorm1 = AddNorm(size=embedding_dim)
        self.masked_attention = nn.MHA(d_model=embedding_dim, n_heads=5)


        pass


    def forward(self, x):
        x = self.embedding(x)

        # create mask here


        
        pass