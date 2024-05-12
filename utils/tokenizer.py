
class Tokenizer:
    def __init__(self, vocab):
        self.vocab = vocab
        self.chars_to_idx = self.tokenize(vocab)
        self.idx_to_chars = self.detokenize(vocab)

    def tokenize(self, vocab):
        chars_to_idx = {ch: i for i, ch in enumerate(vocab)}
        return chars_to_idx

    def detokenize(self, vocab):
        idx_to_chars = {i: ch for i, ch in enumerate(vocab)}
        return idx_to_chars    

    def encode(self, text):
        return [self.chars_to_idx[ch] for ch in text]

    def decode(self, indices):
        return "".join([self.idx_to_chars[i] for i in indices])