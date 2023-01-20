import pickle
import nltk
import flask

import json
import numpy as np
import pandas as pd
import os
import random

from nltk.stem import WordNetLemmatizer

from keras.models import load_model
from diary_management import treat_printing_request

nltk.download('popular')
lemmatizer = WordNetLemmatizer()

nltk.download('stopwords')


model = load_model('./model.tf')
print(model.summary())
data_file = open('./data.json').read()
intents = json.loads(data_file)

words   = pickle.load(open('texts.pkl', 'rb'))
classes = pickle.load(open('labels.pkl', 'rb'))



def preprocess_text(sentence):
    import re

    preprocessed_sentence = sentence

    ## extend contractions
    import contractions

    print(len(contractions.contractions_dict))
    preprocessed_sentence = contractions.fix(preprocessed_sentence)

    ## remove punctuations
    import string
    import sys
    from unicodedata import category

    punctuations = string.punctuation

    punctuation_chars =  [
        chr(i) for i in range(sys.maxunicode)
        if category(chr(i)).startswith("P")
        ]

    punctuations = list(set(punctuation_chars) | set(string.punctuation))
    #punctuations.remove('?')
    print(len(punctuations))

    str_punctuations = ''.join(punctuations)

    preprocessed_sentence = re.sub('[%s]' % re.escape(str_punctuations), '' , preprocessed_sentence)

    ## lower
    preprocessed_sentence = preprocessed_sentence.lower()

    ## remove digits
    preprocessed_sentence = re.sub(r'\w*\d\w*', '', preprocessed_sentence)

    ## remove stop_wrods

    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))

    def remove_stopwords(text):
        return " ".join([word for word in str(text).split() if word not in stop_words])

    preprocessed_sentence = remove_stopwords(preprocessed_sentence)

    ## lemmatization
    """from nltk.stem import WordNetLemmatizer
    
    lemmatizer = WordNetLemmatizer()

    def lemmatize_words(text):
        return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

    preprocessed_sentence = lemmatize_words(preprocessed_sentence)"""

    ## stemming
    from nltk.stem import PorterStemmer
    from nltk.tokenize import word_tokenize

    ps = PorterStemmer()
    words = word_tokenize(preprocessed_sentence)
  
    preprocessed_sentence = ' '.join([ps.stem(word) for word in words])

    ## remove extra spaces
    preprocessed_sentence = re.sub(' +', ' ', preprocessed_sentence)

    return preprocessed_sentence

def predict_printing(model, sentence):
    preprocessed_sentence = preprocess_text(sentence)

    if preprocessed_sentence == '':
        preprocessed_sentence = sentence
    print(preprocessed_sentence)
    return model.predict(np.array(preprocessed_sentence))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    #print("Nonee : ",predict_printing(model, sentence))
    
    #res = predict_printing(model, sentence)[0]
    res=model.predict(np.array([sentence]))[0]
    print(res)
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    target_class = -1
    result = ''
    for i, intent in enumerate(list_of_intents):
        if (intent['tag'] == tag):
            result = random.choice(intent['responses'])
            target_class = i
            break
    return result, target_class

def chatbot_response(msg, model):
    ints = predict_class(msg, model)
    res, target_class = getResponse(ints, intents)
    return res, target_class


from flask import Flask, render_template, request, jsonify
#from flask_cors import CORS

app = Flask(__name__)
#CORS(app)
app.static_folder = 'static'

"""@app.get("/")
def index_get():
    return render_template("index.html")"""

@app.route("/")
def home():
    return render_template("index.html")

"""@app.post("/predict")
def predict():
    text = request.get_json().get("message")
    response, target_class = chatbot_response(text)
    if target_class == -1:
        response = 'error: i didn\'t understand'
    elif target_class == 1:
        treat_printing_request(target_class)
    
    message = {"answer": response}
    return jsonify(message)"""


@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    response, target_class = chatbot_response(userText, model)
    print(response)

    if target_class == -1:
        response = 'error: i didn\'t understand'
    elif target_class == 1:
        (request_treated, error_type) = treat_printing_request(userText)
        if not request_treated:
            if error_type == 0:
                response = 'please precise number of papers and number of pages'
            elif error_type == 1:
                response = 'the printed is not available. It only works between 8pm and 8am'
        else:
            response = 'your printing request has been treated!\nYour documents will be available at ' + error_type
    return response


if __name__ == "__main__":
    app.run()