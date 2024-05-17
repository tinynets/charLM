import torch
from transformer_blocks import Transformer
from utils.tokenizer import Tokenizer
from utils.utils import load_vocab

d_model = 64
num_heads = 8

vocab, vocab_size = load_vocab()
tokenizer = Tokenizer(vocab)

model = Transformer(vocab_size=vocab_size, d_model=d_model, n_decoder_layers=6, n_encoder_layers=6, num_heads=num_heads)
model_path = 'models/model_20240514100506.pth'

state_dict = torch.load(model_path)
model.load_state_dict(state_dict)

input_txt = 'resolve'
start_txt = "we did"

tokenized_input = torch.tensor(tokenizer.encode(input_txt)).unsqueeze(0)
tokenized_start = torch.tensor(tokenizer.encode(start_txt)).unsqueeze(0)



print(f'[INFER SCRIPT]: BEFORE CALLING MODEL')
print(f'tokenized_input shape : {tokenized_input.shape}')
print(f'tokenized_start shape : {tokenized_start.shape}')
print("\n\n")


output = model(tokenized_input, tokenized_start)

predicted_indices = torch.argmax(output, dim=-1)


print(f'indicies : {predicted_indices}')

predicted_tokens = tokenizer.decode(predicted_indices)
print(predicted_tokens)

