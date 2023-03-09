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


batch_size = 32
epoch=5
checkpoint = ModelCheckpoint('sms_spam.h5', 
                             monitor='loss', 
                             verbose=1, 
                             save_best_only=True, mode='min')

history = model.fit(X_train, y_train, epochs=epoch, 
          batch_size=batch_size, verbose=1, callbacks=[checkpoint])