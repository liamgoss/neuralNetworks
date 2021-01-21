import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

data = pd.read_csv("student-mat.csv", sep=";")

data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]  # These are attributes

predict = "G3" # This is our label

#Set up arrays to define attributes and label(s)

x = np.array(data.drop([predict], 1))
y = np.array(data[predict])

# Now let's split these up into 4 variables
# Taking all of our attr. and labels and split them up into 4 arrays
# x train is a section of x, y train is a section of y
# If we trained it off of every single piece of data it would just memorize the pattern - it's already seen the answer
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
'''
# Save the best model we can generate
best = 0
for _ in range(30):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
    # Now let's code a best fit line by creating a training module
    linear = linear_model.LinearRegression()

    linear.fit(x_train, y_train) # find a best fit line using x train and y train
    acc = linear.score(x_test, y_test) # return a value that represents accuracy of our model
    print(acc) # We can determine their grade with about 75% accuracy
    # Accuracy fluctuates with training
    # How do we actually use this model? Lets test it on some data

    # Now let's save our model, since our current model trains super fast it doesn't matter, but in the future we don't
    # want to retrain a model over hundreds of thousands of data sats, plus if we get a hide accuracy model we want it to
    # not fluctuate and keep using it

    # Saving the best
    if acc > best:
        with open("studentmodel.pickle", "wb") as f:
            pickle.dump(linear, f)

'''
# Now read in our pickle file to load this pickle into our linear model
pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)

# y = mx + b will come in handing here
print("Coefficents: ", linear.coef_) # coefficients of our five different variables ~ 5 dimensional space mx + nz + oy ...
print("Intercept: ", linear.intercept_)

# Now how to predict / use on a real student

predictions = linear.predict(x_test)
# We trained it on the train data not the test so this will be untested previously
for i in range(len(predictions)):
    print(predictions[i], x_test[i], y_test[i])
# Prints out (without commas) in the following format:
# Predicted final grade [first grade, second semester grade, study time hours, failures, absences] actual final grade

p = "G1"
style.use("ggplot") # Make our grid look half decent
pyplot.scatter(data[p], data["G3"])# Use a scatter plot
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()