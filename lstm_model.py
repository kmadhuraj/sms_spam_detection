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


def get_lstm_model(config_json, verbose=True):
    model = Sequential()
    model.add(Embedding(config_json['max_features'], 
                      config_json['embedding_len'], 
                      input_length=config_json['input_len']))
    model.add(SpatialDropout1D(0.4))
    model.add(LSTM(config_json['output_lstm_dim'], dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(config_json['classes'], activation='softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer='adam',
                metrics = ['accuracy'])
  
    if verbose:
        print(model.summary())

    return model