import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv1D, Conv2D, Embedding, Dense, LSTM
from tensorflow.keras.layers import GlobalMaxPooling1D, GlobalMaxPooling2D, MaxPooling1D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization, Dropout, Flatten, Input, Reshape

def cnn_nguyen_2_conv2d(x_shape, classes=2):
    model = keras.Sequential([
        Conv2D(16, 3, activation='relu', input_shape=x_shape),
        MaxPooling2D(),
        Conv2D(32, 3, activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(32, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid') if classes < 3 else Dense(classes, activation='softmax')
    ])
    return model

def cnn_nguyen_conv1d_2_conv2d(x_shape, classes=2):
    strides = (x_shape[0] - x_shape[1] + 1, 1) if x_shape[0] > x_shape[1] else (1, x_shape[1] - x_shape[0] + 1)
    model = keras.Sequential([
        Conv2D(16, strides, activation='relu', input_shape=x_shape),
        MaxPooling2D(),
        Conv2D(16, 3, activation='relu'),
        MaxPooling2D(),
        Conv2D(32, 3, activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(32, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid') if classes < 3 else Dense(classes, activation='softmax')
    ])
    return model
    
def cnn_zeng_2_conv2d(x_shape, classes=2):
    model = keras.Sequential([
        Conv2D(16, 3, activation='relu', input_shape=x_shape),
        BatchNormalization(),
        MaxPooling2D(),
        Conv2D(32, 3, activation='relu'),
        BatchNormalization(),
        GlobalMaxPooling2D(),
        Flatten(),
        Dense(32, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid') if classes < 3 else Dense(classes, activation='softmax')
    ])
    return model

def cnn_zeng_3_conv2d(x_shape, classes=2):
    model = keras.Sequential([
        Conv2D(16, 3, activation='relu', input_shape=x_shape),
        BatchNormalization(),
        MaxPooling2D(),
        Conv2D(32, 3, activation='relu'),
        BatchNormalization(),
        MaxPooling2D(),
        Conv2D(64, 3, activation='relu'),
        BatchNormalization(),
        GlobalMaxPooling2D(),
        Flatten(),
        Dense(32, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid') if classes < 3 else Dense(classes, activation='softmax')
    ])
    return model
    
def cnn_zeng_4_conv2d(x_shape, classes=2):
    model = keras.Sequential([
        Conv2D(16, 3, activation='relu', input_shape=x_shape),
        BatchNormalization(),
        MaxPooling2D(),
        Conv2D(32, 3, activation='relu'),
        BatchNormalization(),
        MaxPooling2D(),
        Conv2D(64, 3, activation='relu'),
        BatchNormalization(),
        MaxPooling2D(),
        Conv2D(128, 3, activation='relu'),
        BatchNormalization(),
        GlobalMaxPooling2D(),
        Flatten(),
        Dense(32, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid') if classes < 3 else Dense(classes, activation='softmax')
    ])
    return model

def cnn_zeng_4_conv2d_l2(x_shape, classes=2):
    model = keras.Sequential([
        Conv2D(16, 3, activation='relu', input_shape=x_shape),
        BatchNormalization(),
        MaxPooling2D(),
        Conv2D(32, 3, activation='relu', kernel_regularizer='l2'),
        BatchNormalization(),
        MaxPooling2D(),
        Conv2D(64, 3, activation='relu', kernel_regularizer='l2'),
        BatchNormalization(),
        MaxPooling2D(),
        Conv2D(128, 3, activation='relu', kernel_regularizer='l2'),
        BatchNormalization(),
        GlobalMaxPooling2D(),
        Flatten(),
        Dense(32, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid') if classes < 3 else Dense(classes, activation='softmax')
    ])
    return model

def cnn_deepdbp(x_shape, classes=2):
    model = keras.Sequential([
        keras.Input(shape=(x_shape)),
        Embedding(input_dim=x_shape, output_dim=128),
        Reshape((x_shape, 128, 1)),
        Conv2D(128, (1, 31), padding='same', activation='relu'),
        MaxPooling2D(pool_size=(1, 31)),
        Flatten(),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid') if classes < 3 else Dense(classes, activation='softmax')
    ])
    return model

def deepram_conv1d_onehot(x_shape, classes=2):
    model = keras.Sequential([
        Input(shape=x_shape),
        Dropout(0.5),
        Conv1D(64, 3, activation='relu'),
        MaxPooling1D(),
        Conv1D(128, 3, activation='relu'),
        GlobalMaxPooling1D(),
        Dense(128),
        Dropout(0.5),
        Dense(1, activation='sigmoid') if classes < 3 else Dense(classes, activation='softmax')
    ])
    return model

def deepram_conv1d_embed(x_shape, classes=2):
    model = keras.Sequential([
        Input(shape=(x_shape)),
        Embedding(x_shape, 256),
        Dropout(0.5),
        Conv1D(64, 3, activation='relu'),
        MaxPooling1D(),
        Conv1D(128, 3, activation='relu'),
        GlobalMaxPooling1D(),
        Dense(128),
        Dropout(0.5),
        Dense(1, activation='sigmoid') if classes < 3 else Dense(classes, activation='softmax')
    ])
    return model

def deepram_conv1d_recurrent_onehot(x_shape, classes=2):
    model = keras.Sequential([
        Input(shape=x_shape),
        Dropout(0.5),
        Conv1D(64, 3, activation='relu'),
        MaxPooling1D(),
        Conv1D(128, 3, activation='relu'),
        MaxPooling1D(),
        LSTM(64, return_sequences=True),
        LSTM(128),
        Dense(128),
        Dropout(0.5),
        Dense(1, activation='sigmoid') if classes < 3 else Dense(classes, activation='softmax')
    ])
    return model

def deepram_conv1d_recurrent_embed(x_shape, classes=2):
    model = keras.Sequential([
        Input(shape=(x_shape)),
        Embedding(x_shape, 256),
        Dropout(0.5),
        Conv1D(64, 3, activation='relu'),
        MaxPooling1D(),
        Conv1D(128, 3, activation='relu'),
        MaxPooling1D(),
        LSTM(64, return_sequences=True),
        LSTM(128),
        Dense(128),
        Dropout(0.5),
        Dense(1, activation='sigmoid') if classes < 3 else Dense(classes, activation='softmax')
    ])
    return model

def deepram_recurrent_onehot(x_shape, classes=2):
    model = keras.Sequential([
        Input(shape=x_shape),
        Dropout(0.5),
        LSTM(64, return_sequences=True),
        LSTM(128),
        Dense(128),
        Dropout(0.5),
        Dense(1, activation='sigmoid') if classes < 3 else Dense(classes, activation='softmax')
    ])
    return model

def deepram_recurrent_embed(x_shape, classes=2):
    model = keras.Sequential([
        Input(shape=(x_shape)),
        Embedding(x_shape, 256),
        Dropout(0.5),
        LSTM(64, return_sequences=True),
        LSTM(128),
        Dense(128),
        Dropout(0.5),
        Dense(1, activation='sigmoid') if classes < 3 else Dense(classes, activation='softmax')
    ])
    return model
    
models = {
    'cnn_nguyen_2_conv2d': cnn_nguyen_2_conv2d,
    'cnn_nguyen_conv1d_2_conv2d': cnn_nguyen_conv1d_2_conv2d,
    'cnn_zeng_2_conv2d': cnn_zeng_2_conv2d,
    'cnn_zeng_3_conv2d': cnn_zeng_3_conv2d,
    'cnn_zeng_4_conv2d': cnn_zeng_4_conv2d,
    'cnn_zeng_4_conv2d_l2': cnn_zeng_4_conv2d_l2,
    'cnn_deepdbp': cnn_deepdbp,
    'deepram_conv1d_onehot': deepram_conv1d_onehot,
    'deepram_conv1d_embed': deepram_conv1d_embed,
    'deepram_conv1d_recurrent_onehot': deepram_conv1d_recurrent_onehot,
    'deepram_conv1d_recurrent_embed': deepram_conv1d_recurrent_embed,
    'deepram_recurrent_onehot': deepram_recurrent_onehot,
    'deepram_recurrent_embed': deepram_recurrent_embed,
}

def evaluate(model, history, test_accuracy, y_score, y_true, log_dir, key, dataset, multi_class=False):
    accuracy = pd.DataFrame(history.history)[['accuracy', 'val_accuracy']]
    accuracy = accuracy.max().to_dict()
    accuracy['test_accuracy'] = test_accuracy
    accuracy = pd.DataFrame(accuracy, index=[0])
    accuracy.to_csv(log_dir + f'{key}-{dataset}-accuracy.csv', index=False)
    
    if multi_class:
        roc = pd.DataFrame()
        for cls in range(multi_class):
            fpr, tpr, thresholds = roc_curve(y_true, y_score[:, cls], pos_label=cls)
            temp = pd.DataFrame({
                'ovr': cls,
                'fpr': fpr,
                'tpr': tpr,
                'thresholds': thresholds
            })
            roc = pd.concat([roc, temp])
    else:
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        roc = pd.DataFrame({
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds
        })
    roc.to_csv(log_dir + f'{key}-{dataset}-roc.csv', index=False)
    
    if multi_class:
        pr = pd.DataFrame()
        for cls in range(multi_class):
            precision, recall, thresholds = precision_recall_curve(y_true, y_score[:, cls], pos_label=cls)
            temp = pd.DataFrame({
                'ovr': cls,
                'precision': precision,
                'recall': recall,
                'thresholds': np.append(thresholds, np.nan)
            })
            pr = pd.concat([pr, temp])
    else:
        precision, recall, thresholds = precision_recall_curve(y_true, y_score)
        pr = pd.DataFrame({
            'precision': precision,
            'recall': recall,
            'thresholds': np.append(thresholds, np.nan)
        })
    pr.to_csv(log_dir + f'{key}-{dataset}-pr.csv', index=False)