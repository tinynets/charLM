import torch
import os
from torch.utils.data import DataLoader
from torch import optim
from utils.utils import load_data, create_vocab, preprocess
from utils.tokenizer import Tokenizer
from transformer_blocks import Transformer
from utils.dataset_utils import CharDataset
import datetime


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

model = Transformer(vocab_size=vocab_size, d_model=d_model, n_decoder_layers=6, n_encoder_layers=6, num_heads=num_heads)
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(20):
    # print('epoch:', epoch)
    model.train()
    total_loss = 0 
    for batch, (src, tgt) in enumerate(dataloader):
        optimizer.zero_grad()
        logits = model(src, tgt)


        # TODO: Understand a bit better the shapes and loss calcs
        loss = criterion(logits.view(-1, logits.size(-1)), tgt.view(-1))

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print('epoch:', epoch, 'loss:', total_loss)

torch.save(model.state_dict(), model_output_path)
