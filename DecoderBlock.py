import torch
import torch.nn as nn
from MultiHeadAttention import MHA
from AddNorm import AddNorm
from PositionalEncoding import PositionalEncoding
from PositionwiseFeedForward import PositionwiseFeedForward

class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.attention = MHA(d_model=embedding_dim, n_heads=5)
        self.addnorm1 = AddNorm(size=embedding_dim)
        self.masked_attention = MHA(d_model=embedding_dim, n_heads=5)
        self.pos_enc = PositionalEncoding(d_model=embedding_dim)
        self.mha = MHA(d_model=embedding_dim, n_heads=5)
        self.addnorm2 = AddNorm(size=embedding_dim)
        self.ffn = PositionwiseFeedForward(d_model=embedding_dim, d_ff=embedding_dim*6)
        self.addnorm3 = AddNorm(size=embedding_dim)
        pass

    def create_mask(self, size):
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
        return mask[None, None, :, :]

    def forward(self, x, encoder_output):
        mask = self.create_mask(x.size(1))
        x = self.embedding(x.long())
        x = self.pos_enc(x)
        output, _ = self.masked_attention(x, x, x, mask=mask)
        x = self.addnorm1(x, output)
        cross_attention_output, _ = self.mha(x, encoder_output, encoder_output)
        print(cross_attention_output.size(), 'cross_attention_output size')
        x = self.addnorm2(x, cross_attention_output)
        ffn_output = self.ffn(x)
        x = self.addnorm3(x, ffn_output)
        return x