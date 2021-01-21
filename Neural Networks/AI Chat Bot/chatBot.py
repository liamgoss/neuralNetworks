import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import os
import numpy as np
import tflearn
import tensorflow as tf
from tensorflow.python.framework import ops
import random
import json
import pickle
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

with open("intents.json") as file:
    data = json.load(file)

try:
    #raise AttributeError
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)

except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            # Stemming is getting the roots of the words to train the model on
            # Tokenize means get all the words in out pattern
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds) # wrds is already a list so we use extend
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))  # Convert to list that is sorted and w/o duplicates
    labels = sorted(labels)

    # Neural networks only understand numbers, we need a bag of words (one hot encoded) one hot is 1 if exists otherwise 0
    training = []
    output = []
    # "greeting", "goodbye", "shop", - one hot encode our tags
    out_empty = [0 for _ in range(len(labels))]  # 0 for every tag we have, if the tag is the one we want, then make it a 1
    for x, doc in enumerate(docs_x):
        bag = []
        wrds = [stemmer.stem(w) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)
        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = np.array(training)
    output = np.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

ops.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation='softmax')
net = tflearn.regression(net)

model = tflearn.DNN(net)

try:
    #raise AttributeError
    model.load("model.tflearn")
except:
    model.fit(training, output, n_epoch=250, batch_size=8, show_metric=True)
    model.save("model.tflearn")


def bagOfWords(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = (1)
    return np.array(bag)


def chat():
    print("Start talking with the bot!\nType 'quit' to stop")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break
        results = model.predict([bagOfWords(inp, words)])[0]
        results_index = np.argmax(results)
        tag = labels[results_index]
        print(tag)
        if results[results_index] > 0.7:
            for tg in data["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']
            print(random.choice(responses))
        else:
            print("I don't understand. Please try again or ask a new question!")


chat()
