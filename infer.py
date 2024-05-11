import torch
from Transformer import Transformer
from tokenizer import Tokenizer
from utils import load_and_preprocess_data

vocab_size = 38
d_model = 64
num_heads = 8

data, vocab, vocab_size = load_and_preprocess_data()

tokenizer = Tokenizer(vocab)
model = Transformer(vocab_size=vocab_size, d_model=d_model, n_decoder_layers=6, n_encoder_layers=6, num_heads=num_heads)

state_dict = torch.load('model.pth')
model.load_state_dict(state_dict)

input_txt = 'resolve'
start_txt = "we"

tokenized_input = torch.tensor(tokenizer.encode(input_txt))
tokenized_start_token = torch.tensor(tokenizer.encode(start_txt))


output = model(tokenized_input, tokenized_start_token)

print(output)