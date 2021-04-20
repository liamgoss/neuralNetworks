import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
import PIL
from PIL import Image
import pathlib
import tensorflow as tf
import pickle
from numpy import asarray
from numpy import unique
from numpy import argmax
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.utils import plot_model
from tensorflow.keras.constraints import max_norm
from datetime import datetime
startTime = datetime.now()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
'''
bicycle
bridge
bus
car
chimney
crosswalk
hydrant
motorcycle
other
palm
traffic lights
'''


data_dir = "C:/Users/Liam/Desktop/Programming/Python/tensorEnv/Machine Learning/CaughtCha/Dataset"

batch_size = 32
img_height = 180
img_width = 180

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)



class_names = train_ds.class_names
print(class_names)

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")

for image_batch, labels_batch in train_ds:
    print(image_batch.shape)
    print(image_batch.shape)
    break

AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)  # Cache images into memory after first epoch
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)  # Prefetch overlaps preprocessing and model exec while training

normalization_layer = layers.experimental.preprocessing.Rescaling(1. / 255)  # RGB values aren't ideal for a neural network

num_classes = 11

data_augmentation = keras.Sequential(
    [
        layers.experimental.preprocessing.RandomFlip("horizontal", input_shape=(img_height, img_width, 3)),
        layers.experimental.preprocessing.RandomRotation(0.1),
        layers.experimental.preprocessing.RandomZoom(0.1)
    ]
)
model = Sequential([
        data_augmentation,
        layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(num_classes)
])
'''
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.Dropout(0.2),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.2),
        layers.BatchNormalization(),

        layers.Flatten(),
        layers.Dropout(0.2),
        layers.Dense(256, kernel_constraint=max_norm(3), activation='relu'),
        layers.Dropout(0.2),
        layers.BatchNormalization(),

        layers.Dense(128, kernel_constraint=max_norm(3), activation='relu'),
        layers.Dropout(0.2),
        layers.BatchNormalization(),
        layers.Dense(num_classes, activation='softmax')
'''


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
#plot_model(model, 'newModel.png', show_shapes=True)
print(model.summary())

epochs = 15
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

# Graphs
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# Predict on new data

#imagePath = r"C:\Users\Liam\Desktop\Programming\Python\tensorEnv\Machine Learning\CaughtCha\Test Images\crossing.jpg"
#prediction = model.predict(asarray([imagePath]))
#print('Prediction: class=%d' % argmax(prediction))


directory = r'C:\Users\Liam\Desktop\Programming\Python\tensorEnv\Machine Learning\CaughtCha\Captcha Steals'
guessNames = []
guessPercentages = []
for filename in os.listdir(directory):
    location = "Captcha Steals/" + str(filename)
    img = keras.preprocessing.image.load_img(
        location, target_size=(img_height, img_width)
    )
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    topTwoArgs = np.argpartition(score, -2)[-2:]
    print("-----")
    print(filename)
    photo = mpimg.imread(location)
    imgplot = plt.imshow(photo)
    title = str(class_names[np.argmax(score)]) + " Confidence: " + str(100 * np.max(score))
    plt.title(title)
    plt.show()
    guessNames.append(str(class_names[np.argmax(score)]))
    guessPercentages.append(str(100 * np.max(score)))
    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
            .format(class_names[np.argmax(score)], 100 * np.max(score))
    )

    class_names_Array = np.array(class_names)[topTwoArgs.astype(int)]
    # print("Second Highest Confidence: ", class_names_Array)


print("\n\nPrediction Values Are As Follows...\n")
print(guessNames)
print(guessPercentages)
with open("Scores.txt", "w") as file:
    for i in guessNames:
        j = guessNames.index(i)
        file.write(guessNames[j])
        file.write(", ")
        file.write(guessPercentages[j])
        file.write("\n")
save = input("Do you want to save this model? y / n:")
if save == "y" or save == "Y":
    model.save("CaptchaNetwork")
elif save == "n" or save == "N":
    print("Total Execution Time: ", datetime.now() - startTime)
    quit()
else:
    print("Invalid Input")
    save = input("Do you want to save this model? y / n:")
    if save == "y" or save == "Y":
        model.save("CaptchaNetwork")
        print("Total Execution Time: ", datetime.now() - startTime)
        quit()
    elif save == "n" or save == "N":
        print("Total Execution Time: ", datetime.now() - startTime)
        quit()
    else:
        print("Invalid Input")
        print("Total Execution Time: ", datetime.now() - startTime)
        quit()


