import random
import pickle

random.seed(420)

def load_data(file_name):
    """
        Returns the contents of a text file as a list, of strings, one for each line.
    """
    with open(file_name, 'r') as file:
        lines = file.readlines()

    return lines 

def preprocess(data, shuffle=False):

    if shuffle:
        random.shuffle(data)

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
    padding_token = '~'
    
    vocab = sorted(set(data))
    vocab.append(padding_token)

    with open('vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f)

    return vocab , len(vocab)

def load_vocab():
    with open('vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
        vocab_size = len(vocab)
    return vocab, vocab_size



def split_data(data, splits):
    """
       Splits data into train, val, test splits.
       Splits can be arbitrary, based on proportions provided in splits list.
    """
    total = len(data)
    splits_nums = [int(total * split) for split in splits]
    split_data = []
    start = 0
    for i in range(len(splits_nums) - 1):  # Exclude the last split here
        end = start + splits_nums[i]
        split = data[start:end]
        split_data.append(split)
        start = end
    # For the last split, take all remaining data points
    split_data.append(data[start:])
    return split_data