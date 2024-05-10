import torch
from torch.utils.data import DataLoader
from torch import optim
from utils import load_and_preprocess_data, Tokenizer
from Transformer import Transformer

from dataset_utils import CharDataset

batch_size = 64
d_model = 512
seq_len = 10

data, vocab, vocab_size = load_and_preprocess_data()

# just using 2 batches for now to make sure it goes through the entire network
data = data[:batch_size * 2]

tokenizer = Tokenizer(vocab)


X = []
Y = []

tokenized_data = tokenizer.encode(data)
dataset = CharDataset(data, tokenizer, seq_len)

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

model = Transformer(vocab_size=vocab_size, d_model=d_model, n_decoder_layers=6, n_encoder_layers=6)
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


for batch_idx, (src, tgt) in enumerate(dataloader):
    print(src.shape, tgt.shape)
    

for epoch in range(2):
    print('epoch:', epoch)
    model.train()
    total_loss = 0 
    for batch, (src, tgt) in enumerate(dataloader):
        # print('checking shapes')
        # print(src.shape, tgt.shape)
        
        optimizer.zero_grad()


        logits = model(src, tgt)
        # loss = criterion(logits.transpose(1, 2), Y_batch[:, 1:])

        # loss.backward()
        # optimizer.step()
        # total_loss += loss.item()

    # print('epoch:', epoch, 'loss:', total_loss)