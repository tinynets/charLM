"""
train.py

This script is used to train a Transformer model on a given dataset. It includes functions to create the model 
and to train it. The training function uses the Adam optimizer and CrossEntropyLoss as the loss function.

Functions:
- make_model: Creates a Transformer model with given parameters.
- train_model: Trains the model on a given dataset for a specified number of epochs.

Usage:
- Import this module in your main script and call the functions with appropriate arguments.
"""

import torch
from torch.utils.tensorboard import SummaryWriter
import os
from torch.utils.data import DataLoader
from torch import optim
from utils import load_data, create_vocab, preprocess, split_data, Tokenizer
from transformer_blocks import Transformer
from utils.dataset_utils import CharDataset
import datetime
import time


def make_model(vocab_size, d_model=64, seq_len=10, num_heads=8, device="mps"):
    '''
    Make the model and put it in the gpu and return it
    '''
    model = Transformer(vocab_size=vocab_size, d_model=d_model, n_decoder_layers=6, n_encoder_layers=6, num_heads=num_heads)
    model.to(device)
    return model

def train_model(dataset, model,
    epochs=10, 
    device="mps", 
    batch_size=64, model_name="noname.pth"):

    writer = SummaryWriter()
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    total_batches = len(dataloader)

    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    timestamp = datetime.datetime.now().strftime("%H:%M:%S")

    print(f"Training model {model_name} started at {timestamp}")

    epoch_end_times = []

    

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        model.train()
        total_loss = 0
        start_time = time.time()
        for batch, (src, tgt) in enumerate(dataloader):
            print(f"Batch {batch+1}/{total_batches} of epoch {epoch+1}/{epochs} total_loss: {total_loss}")
            src, tgt = src.to(device), tgt.to(device)
            optimizer.zero_grad()
            logits = model(src, tgt)

            loss = criterion(logits.view(-1, logits.size(-1)), tgt.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            writer.add_scalar("Loss/train", loss.item(), epoch*total_batches + batch)
        writer.flush()
        epoch_time = time.time() - start_time
        epoch_end_times.append(epoch_time)
        print(f"Epoch {epoch+1}/{epochs} loss: {total_loss} time: {epoch_time}")
        
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    model_name = f"{timestamp}_{model_name}.pth"
    save_path = os.path.join("./models", model_name)
    
    
    torch.save(model.state_dict(), save_path)
    print(f"Model saved at {save_path}")

    trained_model = model

    return trained_model

data = load_data("data/input.txt")
preprocessed_data = preprocess(data)
vocab, vocab_size = create_vocab(preprocessed_data)

train_ds, test_ds, val_ds = split_data(preprocessed_data, [0.8, 0.1, 0.1])

# variables needed for making model
d_model = 64
seq_len = 10   
num_heads = 8

model = make_model(vocab_size=vocab_size)

tokenizer = Tokenizer(vocab)

train_ds = CharDataset(train_ds, tokenizer, seq_len)
test_ds = CharDataset(test_ds, tokenizer, seq_len)
val_ds = CharDataset(val_ds, tokenizer, seq_len)


# setup dataset

tokenizer = Tokenizer(vocab)
dataset = CharDataset(preprocessed_data, tokenizer, seq_len)


model = train_model(dataset, model, epochs=2, model_name="transformer.pth", batch_size=128)



