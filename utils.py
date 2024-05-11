def load_data(file_name):
    """
        Returns the contents of a text file as a list, of strings, one for each line.
    """
    with open(file_name, 'r') as file:
        lines = file.readlines()

    return lines 

def preprocess(data):

    data = [line.strip() for line in data] # remove new line char
    data = [line for line in data if line != ""] # remove empties
    data = list(map(str.lower, data)) # make everything lower to reduce vocab
    preprocessed_data = " ".join(data)

    return preprocessed_data

# vocab is needed by itself during inference with additional logic so moving it out of the load_and_preprocess_data function
def create_vocab(data):
    """
        Pass in text, returns a sorted list of unique characters in the text.
        Also returns the size of the vocab.
    """
    
    
    vocab = sorted(set(data))

    return vocab , len(vocab)