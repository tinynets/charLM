def load_and_preprocess_data():
    """
        Returns the data and the vocab (set of unique tokens)
    """

    with open('input.txt', 'r') as file:
        lines = file.readlines()

    lines = [line.strip() for line in lines] # remove new line char
    lines = [line for line in lines if line != ""] # remove empties
    lines = list(map(str.lower, lines)) # make everything lower to reduce vocab

    data = " ".join(lines)
    vocab = sorted(set(data))
    vocab_size = len(vocab)

    return data, vocab, vocab_size 

def tokenize(vocab):
    chars_to_idx = {ch: i for i, ch in enumerate(vocab)}
    return chars_to_idx

def detokenize(vocab):
    idx_to_chars = {i: ch for i, ch in enumerate(vocab)}
    return idx_to_chars


class Tokenizer:
    def __init__(self, vocab):
        self.vocab = vocab
        self.chars_to_idx = tokenize(vocab)
        self.idx_to_chars = detokenize(vocab)

    def encode(self, text):
        return [self.chars_to_idx[ch] for ch in text]

    def decode(self, indices):
        return "".join([self.idx_to_chars[i] for i in indices])