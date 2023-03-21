from flask import Flask, render_template,url_for,request
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
#from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd

import pickle



# create a Flask web application
app = Flask(__name__)

# load the trained LSTM model from a file
model = tf.keras.models.load_model('sms_spam.h5')

# load the tokenizer from a file
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)



# define a route for the home page
@app.route('/')
def home():
    return render_template('home.html')

# define a route for the SMS spam detection form submission
@app.route('/Predict', methods=['POST'])
def predict():
    # retrieve the SMS message from the form data
    message = request.form['message']

    # preprocess the message by tokenizing it and padding it to a fixed length
    sequences = tokenizer.texts_to_sequences([message])
    padded = pad_sequences(sequences, maxlen=160, padding='post', truncating='post')

    # use the LSTM model to predict whether the message is spam or not
    prediction = model.predict(np.array(padded))[0][0]
    if prediction > 0.5:
        result = 'spam'
    else:
        result = 'not spam'

    # return the prediction to the user in a new HTML page
    return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)


