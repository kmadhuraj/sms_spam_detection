spam_data = pd.DataFrame(dataset, columns=['target', 'text'])
spam_data.head(10)


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

only_ham_data = spam_data[spam_data["target"] == "ham"]
only_ham_data.head(10)

only_ham_text = only_ham_data["text"].tolist()
only_spam_text = only_spam_data["text"].tolist()

only_ham_text[0:10]

only_spam_text[0:10]

spam_string = ' '.join(only_spam_text)
ham_string =  ' '.join(only_ham_text)