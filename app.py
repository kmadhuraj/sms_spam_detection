

from flask import Flask, render_template, request,url_for
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
from flask import Response


# create a Flask web application
app = Flask(__name__)


#  load the trained LSTM model from a file
model = tf.keras.models.load_model('sms_spam.h5')




# define a route for the home page  
@app.route('/')
def home():
    return render_template('home.html')

# load the tokenizer from a file
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)
    tokenizer_loaded = tokenizer

# define a route for the SMS spam detection form submission
@app.route('/predict', methods=['POST','GET'])

def predict():
    model = tf.keras.models.load_model('sms_spam.h5')
    global tokenizer_loaded
    if request.method == 'POST':

        message = request.form['message1']
        if message:

            # return render_template('home.html', msg=message)
            
            # return render_template('result.html', seq=message)
            
        
            tokenizer = Tokenizer(num_words=100,oov_token='<00V>')
            tokenizer.fit_on_texts(message)
            word_index=tokenizer.word_index

            sequences = tokenizer.texts_to_sequences(message)

            # return render_template('home.html', seq=sequences)
        
            # maxlen = tokenizer_loaded.get('maxlen')

            padded = pad_sequences(sequences, maxlen=152, padding='post', truncating='post')
            prediction = model.predict(np.array(padded))[0][0]
            
            return render_template('home.html', pred=prediction)
        else:
            return "Please provide a message to classify."
    else:
        return "request is not post"

if __name__ == '__main__':
    app.run(debug=True)













            # tokenizer_loaded=tokenizer
            # if request.method== 'POST':
            #     message = request.form['message1']
                
            #     sequences = tokenizer_loaded.texts_to_sequences([message])
            #     maxlen = tokenizer_loaded.get('maxlen')
            #     padded = pad_sequences(sequences, maxlen=maxlen, padding='post', truncating='post')
        
            
            #     tokenizer_loaded=tokenizer
            #     if tokenizer_loaded is not None:
            #         sequences = tokenizer_loaded.texts_to_sequences([message])
            #         maxlen = tokenizer_loaded.get('maxlen')
            #         padded = pad_sequences(sequences, maxlen=maxlen, padding='post', truncating='post')
            # else:
            #     return ""
            
        
            # prediction = model.predict(np.array(padded))[0][0]

            
            # return render_template('home.html',pred=prediction)


