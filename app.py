

from flask import Flask, render_template, request
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
from flask import Response


# create a Flask web application
app = Flask(__name__)


# # load the trained LSTM model from a file
model = tf.keras.models.load_model('sms_spam.h5')

# load the tokenizer from a file
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)
  

# define a route for the home page
@app.route('/')
def home():
    return render_template('home.html')

# define a route for the SMS spam detection form submission
@app.route('/predict', methods=['GET','POST'])

def predict():
            # retrieve the SMS message from the form 
            # message = request.form.get('message1')
            if request.method== 'POST':
             message = request.form.get('message1')
            message = request.form.get('message1')
            #  print(request.form.get('message'))


            # preprocess the message by tokenizing it and padding it to a fixed length
            tokenizer_loaded = tokenizer
            # tokenizer=None
            if message:
                # tokenizer=Tokenizer([message]) :this code is deleted becouse this line creates new tokenizer 
                # tokenizer.fit_on_texts([message]):this also. but we have to load alredy uploaded tokenizer variable to the method
                # sequences = tokenizer.texts_to_sequences([message])
                # padded = pad_sequences(sequences, maxlen=160, padding='post', truncating='post')
                sequences = tokenizer_loaded.texts_to_sequences([message])
                padded = pad_sequences(sequences, maxlen=tokenizer_loaded.max_len, padding='post', truncating='post')
            else:
                #  padded=None
                 return render_template('home.html', prediction='no message is provided')
                #  result='no message is provided'

            #use the LSTM model to predict whether the message is spam or not
            if tokenizer_loaded is not None:
                prediction = model.predict(np.array(padded))[0][0]
     
                if prediction > 0.5:
                    result = 'spam'
                else:
                    result = 'not spam'
            else:
                 result = 'tokenizer was not initialised'
        
            # return the prediction to the user in a new HTML page
    
            return render_template('home.html',prediction=result)

if __name__ == '__main__':
    app.run(debug=True)

