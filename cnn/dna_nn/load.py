from itertools import product, islice

from Bio import SeqIO
import numpy as np

# generators

## from arrays
def num_records(file):
    ''''count the number of records in fasta file'''
    num = 0
    with open(file, 'r') as f:
        for record in SeqIO.parse(file, 'fasta'):
            num += 1
    return num

def read_fasta(file, limit=None):
    '''
    read fasta file into sequences and labels arrays,
    assuming the names of records are labels
    '''
    with open(file, 'r') as f:
        records = [
            (record.seq._data.upper(), int(record.name)) 
            for record in islice(SeqIO.parse(f, 'fasta'), limit)
        ]
    l = map(list, zip(*records))
    sequences, labels = next(l), next(l)
    print(len(sequences), 'samples loaded')
    return sequences, labels

def gen_from_arrays(sequences, labels, preprocess_funcs=None):
    '''
    create generator from arrays
    preprocess functoins are applied in the order they are supplied
    '''
    def gen():
        for x, y in zip(sequences, labels):
            # multiple funtions
            if preprocess_funcs and isinstance(preprocess_funcs, list):
                for func in preprocess_funcs:
                    x = func(x)
            # one funtion
            elif preprocess_funcs:
                x = preprocess_funcs(x)
            yield x, y
    return gen

# from a large fasta file
def gen_from_fasta(file, preprocess_funcs=None):
    '''
    create generator from a fasta file, assuming the names of records are labels
    preprocess functoins are applied in the order they are supplied
    '''
    def gen():
        f = open(file, 'r')
        for record in SeqIO.parse(f, 'fasta'):
            x = record.seq._data
            # multiple funtions
            if preprocess_funcs and isinstance(preprocess_funcs, list):
                for func in preprocess_funcs:
                    x = func(x)
            # one funtion
            elif preprocess_funcs:
                x = preprocess_funcs(x)
            y = int(record.name)
            yield (x, y)
        f.close()
    return gen

# preprocessing functions

def _check_input(word_size, region_size):
    '''check word_size >= 1 and region_size>= 0'''
    if word_size < 1:
        raise ValueError(f'`word_size` must be at least 1, but got {word_size}')
    if region_size < 0:
        raise ValueError(f'`regions_size` must be at least 0, but got {region_size}')

def encoded_shape(x, word_size=3, region_size=2, onehot=True, expand=True, alphabet='ACGT'):
    '''calculate the shape of encoding base on the sequence length'''
    dim_1 = len(x) - word_size + 1
    dim_2 = ((len(alphabet) ** word_size) if onehot else 1) * (region_size + 1)
    if not region_size and not onehot:
        return (dim_1, 1) if expand else (dim_1,)
    return (dim_1, dim_2, 1) if expand else (dim_1, dim_2)

def decoded_shape(x, word_size=3):
    '''calculate the length of decoded sequence'''
    return len(x) + word_size - 1

def encode(word_size=3, region_size=2, onehot=True, expand=True, alphabet='ACGT'):
    '''transform raw sequences into encoded array'''
    # check input
    _check_input(word_size, region_size)
    # create words (closure)
    words = [''.join(p) for p in product(alphabet, repeat=word_size)]
    word_to_idx = {word: i for i, word in enumerate(words)}
    word_to_idx_func = np.vectorize(lambda word: word_to_idx[word], otypes=[np.int8])
    # actual encoding function
    def encode_func(x):
        # word to index
        if word_size > 1:
            x = [x[i:i+word_size] for i in range(len(x) - word_size + 1)]
        idx = word_to_idx_func(list(x))
        # one-hot encode
        if onehot:
            x = np.zeros((len(idx), len(word_to_idx)))
            x[range(len(idx)), idx] = 1
        else:
            x = idx
        # stack
        # np.hstack does stack along cloumns in the case of 1D array
        stack = np.hstack if len(x.shape) > 1 else np.column_stack
        if region_size:
            x = stack([np.roll(x, -shift, axis=0)
                        for shift in range(region_size + 1)])
        # expand
        if expand:
            x = np.expand_dims(x, axis=-1)
        return x
    return encode_func

def decode(word_size=3, region_size=2, onehot=True, expand=True, alphabet='ACGT'):
    '''transform encoded array back to raw sequences'''
    # check shape
    _check_input(word_size, region_size)
    # create words (closure)
    words = [''.join(p) for p in product(alphabet, repeat=word_size)]
    idx_to_word = {i: word for i, word in enumerate(words)}
    idx_to_word_func = np.vectorize(lambda word: idx_to_word[word], otypes=[np.string_])
    def decode_func(x):
        # first dim is unknown so disregard
        expected_shape = encoded_shape([], word_size, region_size, onehot, expand, alphabet)
        expected_shape = (None,) + expected_shape[1:]
        shape = (None,) + x.shape[1:]
        if shape != expected_shape:
            raise ValueError(f'`x` should have shape {expected_shape}, but got shape {shape}')
        # reverse expand
        if expand:
            x = np.squeeze(x, axis=-1)
        # reverse stack
        if region_size:
            x = x[:, :4**word_size]
        # reverse one-hot encode
        if onehot:
            x = np.argmax(x, -1)
        # reverse word to idx
        x = [idx_to_word[word] for word in x]
        if word_size > 1:
            x = [x[0][:-1]] + [word[-1] for word in x]
        x = ''.join(x)
        return x
    return decode_func

def encode_decode_func(word_size=3, region_size=2, onehot=True, expand=True, alphabet='ACGT'):
    '''return a pair of encode and decode funtions'''
    encode_func = encode(word_size, region_size, onehot, expand, alphabet)
    decode_func = decode(word_size, region_size, onehot, expand, alphabet)
    return encode_func, decode_func