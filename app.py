

from flask import Flask, render_template, request,url_for
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
def home():
    return render_template('home.html')

# load the tokenizer from a file
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)
    # tokenizer_loaded = tokenizer

# define a route for the SMS spam detection form submission
@app.route('/predict', methods=['POST','GET'])





def predict():
    model = tf.keras.models.load_model('sms_spam.h5')
    
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
            cleaned=cleaned_text(message)
            

                
            # return render_template('home.html', msg=message)
            
            # return render_template('result.html', seq=message)
            
            # tokenizer_loaded.fit_on_texts(texts)
            max_features=2000
            delimiter=' '
            tokenizer = Tokenizer(num_words=max_features, split=delimiter)
            # tokenizer = Tokenizer(num_words=200,oov_token='<00V>')
            tokenizer.fit_on_texts(cleaned)
            # here message is tokenized
            # return render_template('result3.html', tok=tokenizer)
            # word_index=to
            # kenizer.word_index

            sequences = tokenizer.texts_to_sequences(cleaned)
            # here the message is converted into sequence of numbers
            # return render_template('result2.html', seq=sequences)
            # return render_template('home.html', seq=sequences)
        
            # maxlen = tokenizer_loaded.get('maxlen')

            padded = pad_sequences(sequences, maxlen=152, padding='post', truncating='post')
            #here sequences are padded into same length
            # return render_template('result.html', pd=padded)
            
            
            array=np.array(padded)

            # return render_template('result2.html', pred=array)
            prediction = model.predict(array)[0][0]
            # prediction = model.predict(np.array(padded))[0][0]
            if prediction >=0.5:
                predicted=1
            else:
                predicted=0
            
            return render_template('home.html', pred=predicted)
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


