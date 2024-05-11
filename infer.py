import torch
from Transformer import Transformer

from utils import load_and_preprocess_data

vocab_size = 38
d_model = 64
num_heads = 8

model = Transformer(vocab_size=vocab_size, d_model=d_model, n_decoder_layers=6, n_encoder_layers=6, num_heads=num_heads)

state_dict = torch.load('model.pth')
model.load_state_dict(state_dict)

input_txt = 'resolve'

print(model.predict(input_txt))