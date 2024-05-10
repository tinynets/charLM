import torch
from torch.utils.data import Dataset


class CharDataset(Dataset):
    def __init__(self, data, tokenizer, seq_len):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.data = self.tokenizer.encode(data)

    def __len__(self):
        return len(self.data) - self.seq_len -1
    
    def __getitem__(self, idx):
        src = self.data[idx:idx + self.seq_len]
        tgt = self.data[idx+1:idx + self.seq_len + 1]

        return torch.tensor(src), torch.tensor(tgt)

