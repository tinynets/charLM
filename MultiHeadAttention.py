import math
import torch
import torch.nn as nn


def scaled_dot_product_attention(q, k, v, mask=None):

    matmul_qk = torch.matmul(q, k.transpose(-1, -2))
    depth = k.size(-1)
    logits = matmul_qk / math.sqrt(depth)

    if mask is not None:
        logits += (mask * -1e9)

    attention_weights = torch.nn.functional.softmax(logits, dim=-1)
    output = torch.matmul(attention_weights, v)

    return output, attention_weights


class MHA(nn.Module):
    def __init__(self, d_model, n_heads=32):
        super(MHA, self).__init__()
        self.n_heads = n_heads
        self.d_model = d_model

        print(f'[MHA] d_model: {self.d_model}')

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)

        self.final_linear = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.n_heads, self.d_model // self.n_heads)
        return x.permute(0, 2, 1, 3)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        query = query.float()
        key = key.float()
        value = value.float()

        print(self.d_model)
        print(query.shape)

        print(self.wq.weight.shape, query.shape)


        q = self.split_heads(self.wq(query), batch_size)
        k = self.split_heads(self.wk(key), batch_size)
        v = self.split_heads(self.wv(value), batch_size)

        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        scaled_attention = scaled_attention.permute(0, 2, 1, 3).contiguous()

        concat_attention = scaled_attention.view(batch_size, -1, self.d_model)
        output = self.final_linear(concat_attention)

        return output, attention_weights
