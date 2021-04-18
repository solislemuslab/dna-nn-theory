import re
import time
import numpy as np
import pandas as pd
from itertools import product

from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from gensim.test.utils import get_tmpfile

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras

def convert_label(row):
  if row["Classes"] == 'EI':
    return 0
  if row["Classes"] == 'IE':
    return 1
  if row["Classes"] == 'N':
    return 2

def n_gram(x, word_size=3):
  arr_x = [c for c in x]
  words = tf.strings.ngrams(arr_x, ngram_width=word_size, separator='').numpy()
  words = list(pd.Series(words).apply(lambda b: b.decode('utf-8')))
  return words

def save_hist(hist, prefix, suffix):
  filename = dir_path+prefix+suffix
  hist_df = pd.DataFrame(hist.history) 
  with open(filename, mode='w') as f:
    hist_df.to_csv(f)

def eval_model(model, x, true_label, ds_name="Training"):
  loss, acc = model.evaluate(x, true_label, verbose=0)
  print("{} Dataset: loss = {} and acccuracy = {}".format(ds_name, np.round(loss, 4), np.round(acc, 4)))
  
# batch_size = 256
SEED = 100
prefix = "uci_doc2vec_"
dir_path = ""
splice_df = pd.read_csv(dir_path+'splice.data', header=None)
splice_df.columns = ['Classes', 'Name', 'Seq']
splice_df["Seq"] = splice_df["Seq"].str.replace(' ', '').str.replace('N', 'A').str.replace('D', 'T').str.replace('S', 'C').str.replace('R', 'G')
splice_df["Label"] = splice_df.apply(lambda row: convert_label(row), axis=1)
splice_df["ngram"] = splice_df.Seq.apply(n_gram)
print('The shape of the datasize is', splice_df.shape)

seq_len = len(splice_df.Seq[0])
print("The length of the sequence is", seq_len)

xtrain_full, xtest, ytrain_full, ytest = train_test_split(splice_df, splice_df.Label, test_size=0.2, random_state=SEED, stratify=splice_df.Label)
xtrain, xval, ytrain, yval = train_test_split(xtrain_full, ytrain_full, test_size=0.2, random_state=SEED, stratify=ytrain_full)
print("shape of training, validation, test set\n", xtrain.shape, xval.shape, xtest.shape, ytrain.shape, yval.shape, ytest.shape)

word_size = 1
vocab = [''.join(p) for p in product('ACGT', repeat=word_size)]


xtrain_ds = tf.data.Dataset.from_tensor_slices((xtrain['Seq'], ytrain)).map(ds_preprocess).batch(batch_size)
xval_ds = tf.data.Dataset.from_tensor_slices((xval['Seq'], yval)).map(ds_preprocess).batch(batch_size)
xtest_ds = tf.data.Dataset.from_tensor_slices((xtest['Seq'], ytest)).map(ds_preprocess).batch(batch_size)

latent_size = 30

model = keras.Sequential([
    keras.Input(shape=(seq_len,)),
    keras.layers.Embedding(seq_len, latent_size),
    keras.layers.LSTM(latent_size, return_sequences=False),
    keras.layers.Dense(128, activation="relu", input_shape=[latent_size]),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(64, activation="relu"),    
    keras.layers.Dropout(0.2),
    keras.layers.Dense(32, activation="relu"),  
    keras.layers.Dropout(0.2), 
    keras.layers.Dense(16, activation="relu"), 
    keras.layers.Dropout(0.2),   
    keras.layers.Dense(3, activation="softmax")                                
])
model.summary()
print(model.summary())
es_cb = keras.callbacks.EarlyStopping(patience=100, restore_best_weights=True)
model.compile(keras.optimizers.Adam(), loss=keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
hist = model.fit(xtrain_ds, validation_data=xval_ds, epochs=4000, callbacks=[es_cb])
path = prefix + '.h5'
model.save(dir_path+path) ## saving  model
save_hist(hist, prefix, "_history.csv")  ## saving acc/loss

ytrain_pred = model.predict(xtrain_ds)
yval_pred = model.predict(xval_ds)
ytest_pred = model.predict(xtest_ds)
res = [ytrain, ytrain_pred, yval, yval_pred, ytest, ytest_pred]
r = 0
for ds in ['train', 'val', 'test']:
  filename = dir_path+prefix + "_" + ds + "_prediction.csv"
  df = pd.DataFrame()
  df[ds] = res[r]
  r += 1
  df['0_prob'] = res[r][:,0]
  df['1_prob'] = res[r][:,1]
  df['2_prob'] = res[r][:,2]  
  r += 1
  with open(filename, mode='w') as f:
      df.to_csv(f)

eval_model(model, xtrain_vec, ytrain, "Training")
eval_model(model, xval_vec, yval, "Validation")
eval_model(model, xtest_vec, ytest, "Test")

