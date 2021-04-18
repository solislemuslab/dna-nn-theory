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
			yield (x_index, y)
		f.close()
	return gen

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
	

def eval_model(model, x, ds_name="Training"):
	loss, acc = model.evaluate(x, verbose=0)
	print("{} Dataset: loss = {} and acccuracy = {}".format(ds_name, np.round(loss, 4), np.round(acc, 4)))
  
batch_size = 256
SEED = 100
prefix = "discovery_baseline_adam256_"
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
train_gen = customize_generator(train_file)
valid_gen = customize_generator(valid_file)
test_gen = customize_generator(test_file)
output_types = (tf.float32, tf.int32)
prefetch = tf.data.experimental.AUTOTUNE
train_ds = Dataset.from_generator(train_gen, output_types=output_types, output_shapes=((101,),())).batch(batch_size).prefetch(prefetch)
valid_ds = Dataset.from_generator(valid_gen, output_types=output_types, output_shapes=((101,),())).batch(batch_size).prefetch(prefetch)
test_ds = Dataset.from_generator(test_gen, output_types=output_types, output_shapes=((101,),())).batch(batch_size).prefetch(prefetch)

latent_size = 30
seq_len = 101
model = keras.Sequential([
    keras.Input(shape=(seq_len,)),
    keras.layers.Embedding(seq_len, latent_size),
    keras.layers.LSTM(latent_size, return_sequences=False),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(64, activation="relu"),    
    keras.layers.Dropout(0.2),
    keras.layers.Dense(32, activation="relu"),  
    keras.layers.Dropout(0.2), 
    keras.layers.Dense(16, activation="relu"), 
    keras.layers.Dropout(0.2),   
    keras.layers.Dense(1, activation="sigmoid")                               
])
print(model.summary())
es_cb = keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
model.compile(keras.optimizers.Adam(), loss=keras.losses.BinaryCrossentropy(), metrics=['accuracy'])
hist = model.fit(train_ds, validation_data=valid_ds, epochs=200, callbacks=[es_cb])
path = prefix + '.h5'
model.save(dir+path) ## saving  model
save_hist(hist, prefix, "_history.csv")  ## saving acc/loss

ytrain_pred = model.predict(train_ds)
yval_pred = model.predict(valid_ds)
ytest_pred = model.predict(test_ds)
ytrain = np.concatenate([y for x, y in train_ds], axis=0)
yval = np.concatenate([y for x, y in valid_ds], axis=0)
ytest = np.concatenate([y for x, y in test_ds], axis=0)
res = [ytrain, ytrain_pred, yval, yval_pred, ytest, ytest_pred]
save_prediction(res, prefix)

eval_model(model, train_ds, "Training")
eval_model(model, valid_ds, "Validation")
eval_model(model, test_ds, "Test")

