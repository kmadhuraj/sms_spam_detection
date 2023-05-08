
import os
os.environ['FLASK_DEBUG'] = 'production'

from flask import Flask, render_template, request,url_for,redirect

import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle


import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

# create a Flask web application
app = Flask(__name__)


#  load the trained LSTM model from a file
model = tf.keras.models.load_model('smsspam2.h5')




# define a route for the home page  

@app.route('/')
def index():
    return render_template('login.html')
@app.route('/login',methods=['POST','GET'])
def login():
        username=request.form['username']
        password=request.form['password']

        if username =='admin' and password =='1234':
            return redirect(url_for('home'))
        else:
            error ='Invalid username or password'
            return render_template('login.html',error=error)
    
    
@app.route('/home')
def home():
    return render_template('index.html')



# define a route for the SMS spam detection form submission
@app.route('/predict', methods=['POST','GET'])

def predict():  
    if request.method == 'POST':
        message = request.form['message1']
        if message:
            def cleaned_text(message):
                message= message.lower()
                message = re.sub(r'[^\w\s]', '', message)
                message = re.sub(r'\d+', '', message)
                message= re.sub(r'http\S+', '', message)
                message = re.sub(r'\S+@\S+', '', message)
                words = nltk.word_tokenize(message)

                stop_words = set(stopwords.words('english'))
                words = [word for word in words if word not in stop_words]
                lemmatizer = WordNetLemmatizer()
                words = [lemmatizer.lemmatize(word) for word in words]

                cleaned_text = ' '.join(words)
                return cleaned_text
            

            
            #setup tokenizer
            max_features=2000
            tokenizer = Tokenizer(num_words=max_features)

            
            #cleaning input message
            cleaned=cleaned_text(message)

            tokenizer.fit_on_texts(cleaned)

            sequences = tokenizer.texts_to_sequences(cleaned)

            maxlen = 152
            padded = pad_sequences(sequences, maxlen=maxlen, padding='post', truncating='post')

            from keras.models import load_model
            model=load_model('./sms_spam.h5')

            prediction = model.predict(padded)[0][0]
            
            if prediction >=0.5:
                predicted=1
            else:
                predicted=0
            
            return render_template('index.html', pred=predicted)
        else:
            return "Please provide a message to classify."
    else:
        return "request is not post"
     

if __name__ == '__main__':
    app.run(debug=True)













       