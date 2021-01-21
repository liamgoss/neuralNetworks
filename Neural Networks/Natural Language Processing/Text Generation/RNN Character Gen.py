#https://stackabuse.com/text-generation-with-python-and-tensorflow-keras/
import numpy as np
import sys
import os
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices: tf.config.experimental.set_memory_growth(device, True)


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
file = open("tifu.txt", encoding="mbcs").read()

def tokenize_words(input):
    input = input.lower()

    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(input)

    # If the created token isn't in the stop words, make it part of "filtered"
    filtered = filter(lambda token: token not in stopwords.words('english'), tokens)
    return " ".join(filtered)

# Preprocess the input data, make tokens
processed_inputs = tokenize_words(file)

chars = sorted(list(set(processed_inputs)))
char_to_num = dict((c, i) for i, c in enumerate(chars))

input_len = len(processed_inputs)
vocab_len = len(chars)

# Now that we have transformed the data into a usable format, we can begin making the dataset out of it
seq_length = 100
x_data = []
y_data = []

for i in range(0, input_len - seq_length, 1):
    # Define input and output sequences
    # Input is the current character plus desired sequence length
    in_seq = processed_inputs[i:i + seq_length]

    # Out sequence is the initial character plus total sequence length
    out_seq = processed_inputs[i + seq_length]

    # We now convert the list of characters to integers
    x_data.append([char_to_num[char] for char in in_seq])
    y_data.append(char_to_num[out_seq])

n_patterns = len(x_data)
print("Total patterns: ", n_patterns)

X = np.reshape(x_data, (n_patterns, seq_length, 1))
X = X/float(vocab_len)

y = to_categorical(y_data)  # One hot encode our label data

# Create our Long Short Term Memory model

batch_size = 32
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
#model.add(LSTM(256, return_sequences=True))
#model.add(Dropout(0.2))
model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')
print(model.summary())
# Set a checkpoint to save the weights then make them the callbacks for out future model
filepath = "model_weights_saved.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
desired_callbacks = [checkpoint]

model.fit(X, y, epochs=200, batch_size=batch_size, callbacks=desired_callbacks)

filename = "model_weights_saved.hdf5"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')

num_to_char = dict((i, c) for i, c in enumerate(chars))

start = np.random.randint(0, len(x_data) - 1)
pattern = x_data[start]
print("Random Seed: ")
print("\"", ''.join([num_to_char[value] for value in pattern]), "\"")

# Now predict what comes next after random seed, num -> char, then append
for i in range(1000):
    x = np.reshape(pattern, (1, len(pattern), 1))
    x = x / float(vocab_len)
    prediction = model.predict(x, verbose=0)
    index = np.argmax(prediction)
    result = num_to_char[index]
    seq_in = [num_to_char[value] for value in pattern]

    sys.stdout.write(result)

    pattern.append(index)
    pattern = pattern[1:len(pattern)]


#Currently overfitting