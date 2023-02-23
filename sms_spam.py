
import os
import re
import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from string import punctuation
from collections import Counter
from keras.layers import LSTM, Dense, Embedding, MaxPooling1D, Conv1D, Dropout, SpatialDropout1D
from keras.models import Sequential
from sklearn.pipeline import Pipeline
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
# %matplotlib inline

def clean(msg):
    for char in ["'", '"', ",", "\n"]:
        msg = msg.replace(char, "")
    return msg

file = open("/spam.csv", encoding="ISO-8859-1")
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

spam_data = pd.DataFrame(dataset, columns=['target', 'text'])
spam_data.head(10)

only_spam_data = spam_data[spam_data["target"] == "spam"]
only_spam_data.head(10)

only_ham_data = spam_data[spam_data["target"] == "ham"]
only_ham_data.head(10)

only_ham_text = only_ham_data["text"].tolist()
only_spam_text = only_spam_data["text"].tolist()

only_ham_text[0:10]

only_spam_text[0:10]

spam_string = ' '.join(only_spam_text)
ham_string =  ' '.join(only_ham_text)

from wordcloud import WordCloud
import matplotlib.pyplot as plt

def generate_plot_wordcloud(dataset, title):
    plt.figure(figsize = (20, 20))
    plt.axis('off')
    plt.imshow(WordCloud(background_color="white", mode="RGBA").generate(dataset))
    plt.title(title)

generate_plot_wordcloud(spam_string, 'Spam Word Clould')

generate_plot_wordcloud(ham_string, 'Ham Word Clould')

import spacy
from spacy import displacy

nlp = spacy.load('en_core_web_sm')

document_spam = nlp(spam_string)
document_ham = nlp(ham_string)

displacy.render(document_spam, style = 'ent', jupyter = True)

displacy.render(document_ham, style = 'ent', jupyter = True)

person_list = [] 
for entity in document_ham.ents:
    if entity.label_ == 'PERSON':
        person_list.append(entity.text)
        
person_list[0: 10]

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

spam_data = preprocessor_dataset_spam(spam_data)
X, y = preprocessor_dataset_tokenizer(spam_data)

print(X.shape, y.shape)

config = {
    'input_len': X.shape[1],
    'max_features': 2000,
    'embedding_len': 128,
    'output_lstm_dim': 200,
    'classes': 2
}

model = get_lstm_model(config)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)

X_train.shape, y_train.shape

batch_size = 32
epoch=5
checkpoint = ModelCheckpoint('sms_spam.h5', 
                             monitor='loss', 
                             verbose=1, 
                             save_best_only=True, mode='min')

history = model.fit(X_train, y_train, epochs=epoch, 
          batch_size=batch_size, verbose=1, callbacks=[checkpoint])

