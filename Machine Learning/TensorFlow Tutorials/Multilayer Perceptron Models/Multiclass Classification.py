#https://machinelearningmastery.com/tensorflow-tutorial-deep-learning-with-tf-keras/
from numpy import argmax
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

# Load the dataset
path = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv'
df = read_csv(path, header=None)
# Split into input and output columns
X, y = df.values[:, :-1], df.values[:, -1]
# Ensure all data are floating point values
X = X.astype("float32")
# Encode strings to integers
y = LabelEncoder().fit_transform(y)
# Split into train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# Determine the number of input features
n_features = X_train.shape[1]
# Define the model
model = Sequential()
model.add(Dense(10, activation='relu', kernel_initializer='he_normal', input_shape=(n_features,)))
model.add(Dense(8, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(3, activation='softmax'))
# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# Fit the model
model.fit(X_train, y_train, epochs=150, batch_size=32, verbose=0)
# Evaluate the model
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print("Test Accuracy: %.3f" % acc)
# Make a predicition
row = [5.1, 3.5, 1.4, 0.2]
yhat = model.predict([row])
print("Predicted: %s (class=%d)" % (yhat, argmax(yhat)))