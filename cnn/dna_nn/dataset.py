import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.data import Dataset

from dna_nn.load import encode, encoded_shape, gen_from_arrays, gen_from_fasta, read_fasta

def splice(file, word_size=3, region_size=0):
    d = {'EI': 0, 'IE': 1, 'N': 2}
    
    data = pd.read_csv(file, header=None, sep=',\\W*', engine='python',
                       usecols=[0, 2])
    data.columns = ['class', 'sequence']
    for old, new in zip('NDSR', 'ATCG'):
        data['sequence'] = data['sequence'].str.replace(old, new)
    data['class'] = data['class'].map(lambda y: d[y])
    
    encode_func = encode(word_size, region_size)
    x_shape = encoded_shape(data['sequence'][0], word_size, region_size)
    
    x, y = data['sequence'].to_numpy(), data['class'].to_numpy()
    x = np.array([encode_func(_) for _ in x])
    x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y)
    return x_shape, x_train, x_test, y_train, y_test

def h3(file, word_size=3, region_size=0):
    sequences, labels = read_fasta(file)
    test_size = 0.15
    val_size = 0.15
    split_options = dict(test_size=test_size, stratify=labels, random_state=3264)
    x_train_val, x_test, y_train_val, y_test = train_test_split(sequences, labels, **split_options)
    # normalize val_size and update options
    split_options.update(dict(test_size=val_size/(1-test_size), stratify=y_train_val))
    x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, **split_options)
    del x_train_val, y_train_val
    
    encode_func = encode(word_size, region_size)
    x_shape = encoded_shape(sequences[0], word_size, region_size)
    
    train_gen = gen_from_arrays(x_train, y_train, encode_func)
    val_gen = gen_from_arrays(x_val, y_val, encode_func)
    test_gen = gen_from_arrays(x_test, y_test, encode_func)
    
    # datasets
    batch_size = 32
    prefetch = tf.data.experimental.AUTOTUNE
    
    output_shapes = (x_shape, ())
    output_types = (tf.float32, tf.float32)
    
    train_ds = Dataset.from_generator(train_gen, output_types, output_shapes)
    train_ds = train_ds.shuffle(500).batch(batch_size).prefetch(prefetch)
    
    test_ds = Dataset.from_generator(test_gen, output_types, output_shapes)
    test_ds = test_ds.batch(batch_size).prefetch(prefetch)
    
    x_val_encode, y_val_encode = [], []
    for x, y in val_gen():
        x_val_encode.append(x)
        y_val_encode.append(y)
    x_val_encode = np.array(x_val_encode)
    y_val_encode = np.array(y_val_encode)
    validation_data = (x_val_encode, y_val_encode)
    
    return x_shape, train_ds, validation_data, test_ds
    
def motif_discovery(train_file, test_file, word_size=3, region_size=2):
    subset_size = 690 * 190
    
    x_shape = encoded_shape(range(101), word_size, region_size)
    encode_func = encode(word_size, region_size)
    train_gen = gen_from_fasta(train_file, encode_func)
    test_gen = gen_from_fasta(test_file, encode_func)
    
    # datasets
    bacth_size = 512
    prefetch = tf.data.experimental.AUTOTUNE
    
    output_shapes = (x_shape, ())
    output_types = (tf.float32, tf.float32)
    
    train_ds = Dataset.from_generator(train_gen, output_types, output_shapes)
    # takes about 30 seconds to skip the training data
    val_ds = train_ds.skip(subset_size).take(690 * 10)
    train_ds = train_ds.take(subset_size).shuffle(500).batch(bacth_size).prefetch(prefetch)
    
    test_ds = Dataset.from_generator(test_gen, output_types, output_shapes)
    test_ds = test_ds.take(subset_size).batch(bacth_size).prefetch(prefetch)
    
    x_val, y_val = [], []
    for d in val_ds:
        x_val.append(d[0])
        y_val.append(d[1])
    x_val = tf.convert_to_tensor(x_val)
    y_val = tf.convert_to_tensor(y_val)
    validation_data = (x_val, y_val)
    
    return x_shape, train_ds, validation_data, test_ds
