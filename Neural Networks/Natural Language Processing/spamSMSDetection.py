import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import json
import os
import numpy as np
import csv
import sys
#752 = old set
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

msgTMP = []
lblTMP = []
filename = r"C:\Users\Liam\Desktop\Programming\Python\tensorEnv\Neural Networks\Natural Language Processing\spam.csv"
with open(filename, newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    for line in spamreader:
        tag = line[:1]
        msg = line[1:2]
        msgTMP.append(msg)
        lblTMP.append(tag)

messages = []
labels = []
# Fix list of 5,000+ lists issue
for i in range(len(msgTMP)):
    #print(lblTMP[i][0], ": ", msgTMP[i][0])
    messages.append(msgTMP[i][0])
    labels.append(lblTMP[i][0])

labelsNumerical = []
# Spam = 1, Ham = 0
for meat in labels:
    if meat == "spam":
        labelsNumerical.append(1)
    elif meat == "ham":
        labelsNumerical.append(0)

#print(labelsNumerical)



training_size = 4179

training_sentences = messages[0:training_size]
testing_sentences = messages[training_size:]

training_labels = labelsNumerical[0:training_size]
testing_labels = labelsNumerical[training_size:]

#print(max(open('spam.csv'), key=len))  # Find the longest line
vocab_size = 10000
max_length = 173
trunc_type = 'post'
padding_type = 'post'
oov_tok = "<OOV>"


tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)

word_index = tokenizer.word_index

training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)



# Convert all variables to numpy arrays
training_padded = np.array(training_padded)
training_labels = np.array(training_labels)
testing_padded = np.array(testing_padded)
testing_labels = np.array(testing_labels)
#print(training_labels)
embedding_dim = 16

model = tf.keras.Sequential([
    # Adding an embedding layer for the neural network to learn the vectors
    layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    # Global Averaging pooling is similar to adding up vectors in this case
    layers.GlobalAveragePooling1D(),
    layers.Dense(24, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

num_epochs = 20

try:
    #raise AttributeError
    #model = load_model("savedModel")
    model = load_model(r"C:\Users\Liam\Desktop\Programming\Python\tensorEnv\Neural Networks\Natural Language Processing\savedModel")
    #print("-----\nLoading Model...\n-----")

except:
    #print("-----\nTraining Model...\n-----")
    history = model.fit(training_padded, training_labels, epochs=num_epochs,
                        validation_data=(testing_padded, testing_labels))
    #print("Saving Model....")
    #model.save("savedModel")
    model.save(r"C:\Users\Liam\Desktop\Programming\Python\tensorEnv\Neural Networks\Natural Language Processing\savedModel")

#model.load("savedModel")
'''
# Sentence 1 is spam, sentence 2 is ham (legit)
new_sentence = [
    "As a valued customer, I must inform you that your extended warranty will be awarded with a 2 year extension, call 18007392839",
    "CBD COFFEE a powerful natural relief! Proven to reduce pain, aches, anxiety and stress. Get yours now! Reply STOP to opt out",
    "Hey how are you doing?",
    "I wanna be in the room where it happens. No one else was in the room where it happens",
    "It's the same old theme in 2018, in your head (x2) they're still fighting"
]

new_sequences = tokenizer.texts_to_sequences(new_sentence)
new_padded = pad_sequences(new_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
new_padded = np.array(new_padded)


values = model.predict(new_padded)
i = 0
for _ in enumerate(values):
    try:
        percentage = values[i]
        if "e" in str(percentage):
            newval = "{:.7f}".format(percentage[0])
            #print(newval)
        else:
            pass
            #print(percentage[0])
    except IndexError:
        break
    i = i + 1
'''
def isSpam(userInput):
    userSentence = []
    userSentence.append(str(userInput))
    user_sequences = tokenizer.texts_to_sequences(userSentence)
    user_padded = pad_sequences(user_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
    user_padded = np.array(user_padded)

    userPrediction = model.predict(user_padded)
    #print(userPrediction[0])
    response = "This message is {:.2f}% likely to be spam.".format(userPrediction[0][0] * 100)
    return response

print(isSpam(str(sys.argv[1])))

