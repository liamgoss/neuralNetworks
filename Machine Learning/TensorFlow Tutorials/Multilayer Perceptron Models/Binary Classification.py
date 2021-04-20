# https://machinelearningmastery.com/tensorflow-tutorial-deep-learning-with-tf-keras/

# The three most common loss functions are:
# ‘binary_crossentropy‘ for binary classification.
# ‘sparse_categorical_crossentropy‘ for multi-class classification.
# ‘mse‘ (mean squared error) for regression.

import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load the dataset
path = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/ionosphere.csv"
df = read_csv(path, header=None)
# Split into two input and output columns
X, y = df.values[:, :-1], df.values[:, -1]
# Ensure all data are floats
X = X.astype("float32")
# Encode strings to integers
y = LabelEncoder().fit_transform(y)
# Split into train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# Determine the number of input features
n_features = X_train.shape[1]
# Define the model
model = Sequential()
model.add(Dense(10, activation='relu', kernel_initializer='he_normal', input_shape=(n_features,)))
model.add(Dense(8, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(1, activation='sigmoid'))
# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# Fit the model
model.fit(X_train, y_train, epochs=150, batch_size=32, verbose=2)
# Evaluate the model
loss, acc = model.evaluate(X_test, y_test, verbose=2)
print("Test Accuracy: %.3f" % acc)
# Make a prediction
row = [1,0,0.99539,-0.05889,0.85243,0.02306,0.83398,-0.37708,1,0.03760,0.85243,-0.17755,0.59755,-0.44945,0.60536,-0.38223,0.84356,-0.38542,0.58212,-0.32192,0.56971,-0.29674,0.36946,-0.47357,0.56811,-0.51171,0.41078,-0.46168,0.21266,-0.34090,0.42267,-0.54487,0.18641,-0.45300]
yhat = model.predict([row])
print('Predicted: %.3f' % yhat)
