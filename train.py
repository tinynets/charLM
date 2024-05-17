import torch
import os
from torch.utils.data import DataLoader
from torch import optim
from utils.utils import load_data, create_vocab, preprocess
from utils.tokenizer import Tokenizer
from transformer_blocks import Transformer
from utils.dataset_utils import CharDataset
import datetime
import time


device = torch.device("mps")
if str(device) != "mps":
    raise ValueError("Invalid device. Only 'mps' is supported.")

timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
model_output_path = f'models/model_{timestamp}.pth'


batch_size = 64
d_model = 64
seq_len = 10
num_heads = 8


data = load_data("data/input.txt")
preprocessed_data = preprocess(data)
vocab, vocab_size = create_vocab(preprocessed_data)

# just using 2 batches for now to make sure it goes through the entire network
preprocessed_data = preprocessed_data[:batch_size * 2]


tokenizer = Tokenizer(vocab)

X = []
Y = []

dataset = CharDataset(preprocessed_data, tokenizer, seq_len)

tokenized_data = tokenizer.encode(preprocessed_data)

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)


# some logic to see what is inside a batch
for batch_idx, (src, tgt) in enumerate(dataloader):
    if batch_idx < 2:
        print(f"Batch {batch_idx + 1}:")
        print("Source:", src.shape)
        print("Target:", tgt.shape)
        # print(src[0])
    else:
        break


model = Transformer(vocab_size=vocab_size, d_model=d_model, n_decoder_layers=6, n_encoder_layers=6, num_heads=num_heads)
model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

total_batches = len(dataloader)


start_time = time.time()    
epoch_times = []

for epoch in range(2):
    print('epoch:', epoch)
    model.train()
    total_loss = 0 
    for batch, (src, tgt) in enumerate(dataloader):
        # print(f'batch: {batch}/{total_batches}')
        src, tgt = src.to(device), tgt.to(device)
        optimizer.zero_grad()
        logits = model(src, tgt)


        # TODO: Understand a bit better the shapes and loss calcs
        loss = criterion(logits.view(-1, logits.size(-1)), tgt.view(-1))

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    epoch_time = time.time() - start_time
    epoch_times.append(epoch_time)
    print('epoch:', epoch, 'loss:', total_loss, 'time:', epoch_time)

torch.save(model.state_dict(), model_output_path)
