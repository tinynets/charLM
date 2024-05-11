import torch
from torch.utils.data import DataLoader
from torch import optim
from utils import load_and_preprocess_data
from tokenizer import Tokenizer
from Transformer import Transformer

from dataset_utils import CharDataset

batch_size = 64
d_model = 64
seq_len = 10
num_heads = 8

data, vocab, vocab_size = load_and_preprocess_data()

# just using 2 batches for now to make sure it goes through the entire network
data = data[:batch_size * 2]

tokenizer = Tokenizer(vocab)

X = []
Y = []

tokenized_data = tokenizer.encode(data)
dataset = CharDataset(data, tokenizer, seq_len)

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

torch.save(model.state_dict(), 'model.pth')
