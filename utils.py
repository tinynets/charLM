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



