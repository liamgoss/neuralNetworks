#https://towardsdatascience.com/natural-language-processing-with-tensorflow-e0a701ef5cef
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
import os
import numpy as np

f = open("Sarcasm_Headlines_Dataset.json", 'r')
data = []
for line in f.readlines():
    dic = json.loads(line)
    data.append(dic)
f.close()
print(data)

sentences = []
labels = []
urls = []

for item in data:
    sentences.append(item['headline'])
    labels.append(item['is_sarcastic'])
    urls.append(item['article_link'])

training_size = 20000

training_sentences = sentences[0:training_size]
testing_sentences = sentences[training_size:]

training_labels = labels[0:training_size]
testing_labels = labels[training_size:]

vocab_size = 10000
max_length = 100
trunc_type = 'post'
padding_type = 'post'
oov_tok = "<OOV>"
training_size = 20000

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

num_epochs = 10

history = model.fit(training_padded, training_labels, epochs=num_epochs,
                    validation_data=(testing_padded, testing_labels))

# Sentence 1 is kinda sarcastic, sentence 2 is a statement
new_sentence = [
    "granny starting to fear spider in the garden might be real",
    "game of thrones season finale showing this sunday night"
]

new_sequences = tokenizer.texts_to_sequences(new_sentence)
new_padded = pad_sequences(new_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
new_padded = np.array(new_padded)

print(model.predict(new_padded))





