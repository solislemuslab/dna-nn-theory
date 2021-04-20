import re
import time
import numpy as np
import pandas as pd
from itertools import product

import sklearn.metrics as metrics

import tensorflow as tf
from tensorflow import keras
from tensorflow.data import Dataset

def customize_generator(file):
	def gen():
		f = open(file, 'r')
		for line in f:
			line = line.strip()
			x = line[:-2]
			x_index = tf.subtract(create1gram(x), 2)
			y = int(line[-1])
			yield (x_index, x_index)
		f.close()
	return gen

def get_label(file):
	labels = []
	f = open(file, 'r')
	for line in f:
		line = line.strip()
		y = int(line[-1])
		labels.append(y)
	return np.array(labels)
				

def save_hist(hist, prefix, suffix):
	filename = prefix+suffix
	hist_df = pd.DataFrame(hist.history) 
	with open(filename, mode='w') as f:
		hist_df.to_csv(f)

def save_prediction(res, prefix):
	ds_order = ['train', 'val', 'test']
	for idx in range(0,6,2):
		label = res[idx]
		pred = res[idx+1]
		ds = ds_order[int(idx/2)]
		## save true label and prediction
		pred_filename = prefix + ds + "_prediction.csv"
		df_pred = pd.DataFrame()
		df_pred[ds] = label
		df_pred[ds+'_pred'] = pred
		df_pred.to_csv(pred_filename)
		## save fpr & tpr for roc
		roc_filename = prefix + ds + "_roc.csv"
		fpr, tpr, thresholds = metrics.roc_curve(label, pred)
		df_roc = pd.DataFrame({
			'fpr': fpr,
			'tpr': tpr,
			'thresholds': thresholds
		})
		df_roc.to_csv(roc_filename)
		## save precision and recall 
		pr_filename = prefix + ds + "_pr.csv"
		precision, recall, thresholds = metrics.precision_recall_curve(label, pred)
		df_pr = pd.DataFrame({
			'precision': precision,
			'recall': recall,
		})
		df_pr.to_csv(pr_filename)
	

def eval_model(model, x, true_label, ds_name="Training"):
	loss, acc = model.evaluate(x, true_label, verbose=0)
	print("{} Dataset: loss = {} and acccuracy = {}".format(ds_name, np.round(loss, 4), np.round(acc, 4)))
  
batch_size = 256
prefix = "discovery_ae_adam256_"
dir = ""
validation_size = 690 * 10
subset_size = 690 * 139

word_size = 1
neucleotides = 'ACGT'
vocab = [''.join(p) for p in product(neucleotides, repeat=word_size)]
vocab_size = len(neucleotides)
print('vocab_size:', vocab_size)
create1gram = keras.layers.experimental.preprocessing.TextVectorization(
	standardize=lambda x: tf.strings.regex_replace(x, '(.)', '\\1 '), ngrams=1
)
create1gram.adapt(vocab)

## reading data
train_file = dir + 'motif_discovery-train.txt'
valid_file = dir + 'motif_discovery-valid.txt'
test_file = dir + 'motif_discovery-test.txt'
ytrain = get_label(train_file)
yval = get_label(valid_file)
ytest = get_label(test_file)

train_gen = customize_generator(train_file)
valid_gen = customize_generator(valid_file)
test_gen = customize_generator(test_file)
output_types = (tf.float32, tf.float32)
prefetch = tf.data.experimental.AUTOTUNE
xtrain_seq = Dataset.from_generator(train_gen, output_types=output_types, output_shapes=((101,),(101,))).batch(batch_size).prefetch(prefetch)
xval_seq = Dataset.from_generator(valid_gen, output_types=output_types, output_shapes=((101,),(101,))).batch(batch_size).prefetch(prefetch)
xtest_seq = Dataset.from_generator(test_gen, output_types=output_types, output_shapes=((101,),(101,))).batch(batch_size).prefetch(prefetch)

latent_size = 30
seq_len = 101
encoder = keras.Sequential([
    keras.Input(shape=(seq_len,)),
    keras.layers.Embedding(seq_len, latent_size),
    keras.layers.LSTM(latent_size, return_sequences=False),
])

decoder = keras.Sequential([
    keras.layers.RepeatVector(seq_len, input_shape=[latent_size]),
    keras.layers.LSTM(latent_size, return_sequences=True),
    keras.layers.TimeDistributed(keras.layers.Dense(4, activation='softmax'))  # ACTG
])

recurrent_ae = keras.Sequential([encoder, decoder])
print(recurrent_ae.summary())
es_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
recurrent_ae.compile(optimizer=tf.optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics='accuracy')
ae_hist = recurrent_ae.fit(xtrain_seq, validation_data=xval_seq, epochs=200, callbacks=[es_cb])
path = prefix + '_recurrent.h5'
recurrent_ae.save(dir+path) ## saving  model
save_hist(ae_hist, prefix, "_reconstruction_history.csv")  ## saving acc/loss

xtrain_vec = encoder.predict(xtrain_seq)
xval_vec = encoder.predict(xval_seq)
xtest_vec = encoder.predict(xtest_seq)
print('The shape of xtrain/xval/xtest_seq is', xtrain_vec.shape, xval_vec.shape, xtest_vec.shape)

model = keras.models.Sequential([
  keras.layers.Dense(128, activation="relu", input_shape=[latent_size]),
  keras.layers.Dropout(0.2),
  keras.layers.Dense(64, activation="relu"),    
  keras.layers.Dropout(0.2),
  keras.layers.Dense(32, activation="relu"),  
  keras.layers.Dropout(0.2), 
  keras.layers.Dense(16, activation="relu"), 
  keras.layers.Dropout(0.2),   
  keras.layers.Dense(1, activation="sigmoid")                               
])
model.compile(keras.optimizers.SGD(learning_rate=0.001, momentum=0.9), loss=keras.losses.BinaryCrossentropy(), metrics=['accuracy'])
model_hist = model.fit(xtrain_vec, ytrain, validation_data=(xval_vec, yval), epochs=500, callbacks=[es_cb])
path = prefix + '_prediction.h5'
model.save(dir+path) ## saving  model
save_hist(model_hist, prefix, "_prediction_history.csv")  ## saving acc/loss

ytrain_pred = model.predict(xtrain_vec)
yval_pred = model.predict(xval_vec)
ytest_pred = model.predict(xtest_vec)
res = [ytrain, ytrain_pred, yval, yval_pred, ytest, ytest_pred]
save_prediction(res, prefix)

eval_model(model, xtrain_vec, ytrain, "Training")
eval_model(model, xval_vec, yval,"Validation")
eval_model(model, xtest_vec, ytest,"Test")

