import json
from flask import Flask, jsonify, request
import random
import numpy as np
import pandas as pd
import nltk
import tensorflow as tf
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()


app = Flask(__name__)

@app.post("/bot")
def bot():
    with open('intents.json') as json_data:
        intents = json.load(json_data)
    words = []
    classes = []
    documents = []
    ignore_words = ['?']
    # loop through each sentence in the intent's patterns
    for intent in intents['intents']:
        for pattern in intent['patterns']:
            # tokenize each and every word in the sentence
            w = nltk.word_tokenize(pattern)
            # add word to the words list
            words.extend(w)
            # add word(s) to documents
            documents.append((w, intent['tag']))
            # add tags to our classes list
            if intent['tag'] not in classes:
                classes.append(intent['tag'])
    # stem and lower each word and remove duplicates
    words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
    words = sorted(list(set(words)))
    # sort classes
    classes = sorted(list(set(classes)))
    # load model
    model = tf.keras.models.load_model('bot_model.h5')
    

    def clean_up_sentence(sentence):
        # tokenize the pattern - split words into array
        sentence_words = nltk.word_tokenize(sentence)
        # stem each word - create short form for word
        sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
        return sentence_words
    # return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
    def bow(sentence, words):
        # tokenize the pattern
        sentence_words = clean_up_sentence(sentence)
        # bag of words - matrix of N words, vocabulary matrix
        bag = [0]*len(words)  
        for s in sentence_words:
            for i,w in enumerate(words):
                if w == s: 
                    # assign 1 if current word is in the vocabulary position
                    bag[i] = 1
        return(np.array(bag))


    def classify(sentence):
        ERROR_THRESHOLD = 0.25
    
        input_data = pd.DataFrame([bow(sentence, words)], dtype=float, index=['input'])
        results = model.predict([input_data])[0]
        # filter out predictions below a threshold, and provide intent index
        results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
        # sort by strength of probability
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append((classes[r[0]], str(r[1])))
        # return tuple of intent and probability
    
        return return_list

    def response(sentence):
        results = classify(sentence)
        # if we have a classification then find the matching intent tag
        if results:
            # loop as long as there are matches to process
            while results:
                for i in intents['intents']:
                    # find a tag matching the first result
                    if i['tag'] == results[0][0]:
                        # a random response from the intent
                        return random.choice(i['responses'])

                results.pop(0)
        else:
            return "I didn't get that"

    res = response(request.form["query"])
    return jsonify({"data": res})

@app.route("/")
def hello_world():
    return "Welcome to Dokta!"

if __name__ == "__main__":
    app.run(host="0.0.0.0")