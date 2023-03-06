import os
import re
import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from string import punctuation
from collections import Counter
from keras.layers import LSTM, Dense, Embedding, MaxPooling1D, Conv1D, Dropout, SpatialDropout1D
from keras.models import Sequential
from sklearn.pipeline import Pipeline
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
%matplotlib inline

def preprocessor_dataset_spam(spam_data):
    spam_data['text'] = spam_data['text'].apply(lambda x: str(x).lower())
    spam_data['text'] = spam_data['text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))
    spam_data['target'] = spam_data['target'].map({'ham': 0, 'spam': 1})
    spam_data.head(10)
    return spam_data

def preprocessor_dataset_tokenizer(spam_data, max_features=2000, delimiter=' '):
    tokenizer = Tokenizer(num_words=max_features, split=delimiter)
    tokenizer.fit_on_texts(spam_data['text'].values)
    X = pad_sequences(tokenizer.texts_to_sequences(spam_data['text'].values))
    y = pd.get_dummies(spam_data['target']).values
    return X, y