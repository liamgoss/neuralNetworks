import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing
# Preprocessing is for converting the noninteger values into mediums (0, 1, 2, 3, etc)
# This is a classification algorithm
# Tries to classify a data point based on known classes (predict what group a point belongs to basically)
# k is the amount of neighbors we are going to look for
# Uses the distance formula to find distance to *every* neighboring point - can be resource intensive in large data sets

data = pd.read_csv("car.data")
print(data.head())

le = preprocessing.LabelEncoder()
# Create a list for each column in the data
buying = le.fit_transform(data["buying"]) # Gets entire buying column into a list and transform into appr. int. values
maint = le.fit_transform(data["maint"])
door = le.fit_transform(data["door"])
persons = le.fit_transform(data["persons"])
lug_boot = le.fit_transform(data["lug_boot"])
safety = le.fit_transform(data["safety"])
cls = le.fit_transform(data["class"])

predict = "class"

x = list(zip(buying, maint, door, persons, lug_boot, safety))
y = list(cls)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)


model = KNeighborsClassifier(n_neighbors=7)
model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
print(acc)
predicted = model.predict(x_test)
names = ["unacc", "acc", "good", "vgood"]
for i in range(len(predicted)):
    print("Predicted:", names[predicted[i]], "  Data: ", x_test[i], " Actual: ", names[y_test[i]])
    n = model.kneighbors([x_test[i]], 7, True)
    print("N: ", n, "\n")