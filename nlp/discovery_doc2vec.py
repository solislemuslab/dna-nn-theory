import re
import time
import numpy as np
import pandas as pd
from itertools import product

import sklearn.metrics as metrics

import tensorflow as tf
from tensorflow import keras
from tensorflow.data import Dataset

import multiprocessing
from gensim.models import Doc2Vec
from gensim.models.doc2vec import Doc2Vec,TaggedDocument
from gensim.test.utils import get_tmpfile

def getVec(model, tagged_docs, epochs=20):
	sents = tagged_docs.values
	regressors = [model.infer_vector(doc.words, epochs=epochs) for doc in sents]
	return np.array(regressors)

def doc2vec_training(embed_size_list=[50,100,150,200], figsize=(10,50), verbose=0):
  num_model = len(embed_size_list)
  counter = 0
  model_list = []
  hist_list = []
  es_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
  for embed_size in embed_size_list:
    start = time.time()
    print("training doc2vec for embedding size =", embed_size)
    model_dm = Doc2Vec(dm=1, vector_size=embed_size, negative=5, hs=0, min_count=2, sample=0, workers=cores)
    if verbose == 1:
      model_dm.build_vocab([x for x in xtrain_tagged.values])
    else:
      model_dm.build_vocab(xtrain_tagged.values)

    for epoch in range(80):
      if verbose == 1:
        model_dm.train([x for x in xtrain_tagged.values], \
				  total_examples=len(xtrain_tagged.values), epochs=1)
      else:
        model_dm.train(xtrain_tagged.values, total_examples=len(xtrain_tagged.values), epochs=1)
      model_dm.alpha -= 0.002
      model_dm.min_alpha = model_dm.alpha
    xtrain_vec = getVec(model_dm, xtrain_tagged)
    xval_vec = getVec(model_dm, xval_tagged)
    xtest_vec = getVec(model_dm, xtest_tagged)
		# save the embedding to csv files
    train_filename = "size" + str(embed_size) + "_train.csv"
    val_filename = "size" + str(embed_size) + "_val.csv"
    test_filename = "size" + str(embed_size) + "_test.csv"
    np.savetxt(prefix + train_filename, xtrain_vec, delimiter=",")
    np.savetxt(prefix + val_filename, xval_vec, delimiter=",")
    np.savetxt(prefix + test_filename, xtest_vec, delimiter=",")
    print("the shape for training vector is", xtrain_vec.shape, \
		  "the shape for val vector is", xval_vec.shape, \
		  "the shape for test vector is", xtest_vec.shape)
    counter += 1

    print("embedding size =", embed_size)
    model = keras.Sequential([
			keras.layers.Dense(128, activation="relu", input_shape=[embed_size]),
			keras.layers.Dropout(0.2),
			keras.layers.Dense(64, activation="relu"), 
			keras.layers.Dropout(0.2),
			keras.layers.Dense(32, activation="relu"), 
			keras.layers.Dropout(0.2),
			keras.layers.Dense(16, activation="relu"), 
			keras.layers.Dropout(0.2),
			keras.layers.Dense(1, activation="sigmoid")                        
		])
    model.compile(keras.optimizers.SGD(momentum=0.9), \
			  "binary_crossentropy", metrics=["accuracy"])
    hist = model.fit(xtrain_vec, ytrain, \
			  epochs=400, callbacks=[es_cb], validation_data=(xval_vec, yval))
    train_loss, train_acc = model.evaluate(xtrain_vec, ytrain)
    val_loss, val_acc = model.evaluate(xval_vec, yval)
    test_loss, test_acc = model.evaluate(xtest_vec, ytest)
    print("Evaluation on training set: loss", train_loss, \
		  "accuracy", train_acc)
    print("Evaluation on val set: loss", val_loss, \
		  "accuracy", val_acc)
    print("Evaluation on test set: loss", test_loss, \
			  "accuracy", test_acc)
    model_list.append(model)
    model.save(prefix+str(embed_size)+"_"+DATE+".h5")
    hist_list.append(hist)
    save_hist(hist, prefix, "size"+str(embed_size)+".csv" )
		
    ytrain_pred = model.predict(xtrain_vec)
    yval_pred = model.predict(xval_vec)
    ytest_pred = model.predict(xtest_vec)
    res = [ytrain, ytrain_pred, yval, yval_pred, ytest, ytest_pred]
    save_prediction(res, prefix+"size"+str(embed_size))
    end = time.time()
    print("running time in ", end - start, "seconds")
    print("\n\n")
  return model_list, hist_list


def read_data(file):
	df = pd.read_csv(file, sep=",", header=None, names=["seq","label"])		
	return df
	
def preprocess(df):
	df['ngram'] = df['seq'].apply(n_gram)
	return df

def n_gram(x, word_size=3):
	arr_x = [c for c in x]
	words = tf.strings.ngrams(arr_x, ngram_width=word_size, separator='').numpy()
	words = list(pd.Series(words).apply(lambda b: b.decode('utf-8')))
	return words

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
prefix = "discovery_doc2vec_"
dir = ""
validation_size = 690 * 10
subset_size = 690 * 139
DATE = "20210413"

## reading data
train_file = dir + 'motif_discovery-train.txt'
valid_file = dir + 'motif_discovery-valid.txt'
test_file = dir + 'motif_discovery-test.txt'
df_train = read_data(train_file)
df_val = read_data(valid_file)
df_test = read_data(test_file)
df_train = preprocess(df_train)
df_val = preprocess(df_val)
df_test = preprocess(df_test)

ytrain = df_train['label']
yval = df_val['label']
ytest = df_test['label']
cores = multiprocessing.cpu_count()

word_size = 3
vocab = [''.join(c) for c in product('ATCG', repeat=word_size)]

xtrain_tagged = df_train.apply(
    lambda r: TaggedDocument(words=r["ngram"], tags=[r["label"]]), axis=1
)
xval_tagged = df_val.apply(
    lambda r: TaggedDocument(words=r["ngram"], tags=[r["label"]]), axis=1
)
xtest_tagged = df_test.apply(
    lambda r: TaggedDocument(words=r["ngram"], tags=[r["label"]]), axis=1
)

## training
embed_size_list = [50]
num_model = len(embed_size_list)
model_list, hist_list = doc2vec_training(embed_size_list)	
