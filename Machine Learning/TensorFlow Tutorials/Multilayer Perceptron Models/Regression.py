#https://machinelearningmastery.com/tensorflow-tutorial-deep-learning-with-tf-keras/
from numpy import sqrt
from pandas import read_csv
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

# Load the dataset
path = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv'
df = read_csv(path, header=None)
# Split into input and output columns
X, y = df.values[:, :-1], df.values[:, -1]
# Split into train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# Determine the number of input features
n_features = X_train.shape[1]
# Define the model
model = Sequential()
model.add(Dense(10, activation='relu', kernel_initializer='he_normal', input_shape=(n_features,)))
model.add(Dense(8, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(1))
# Compile the model
model.compile(optimizer='adam', loss='mse')
# Fit the model
model.fit(X_train, y_train, epochs=150, batch_size=32, verbose=0)
# Evaluate the model
error = model.evaluate(X_test, y_test, verbose=0)
print("MSE: %.3f, RMSE: %.3f" % (error, sqrt(error)))
# Make a prediction
row = [0.00632,18.00,2.310,0,0.5380,6.5750,65.20,4.0900,1,296.0,15.30,396.90,4.98]
yhat = model.predict([row])
print('Predicted: %.3f' % yhat)