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

def clean(msg):
    for char in["'",'"',",","\n"]:
      msg=msg.replace(char,"")
    return msg

file = open("/content/spam.csv", encoding="ISO-8859-1")
dataset = list()
for i, current in enumerate(file):
    if i > 0:
        df_spam = current.split(',')
        msgs = df_spam[1:]
        msgs_str = ''
        for msg in msgs:
            msgs_str += clean(msg)
        target = df_spam[0]
        dataset.append([target, msgs_str])