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
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import plot_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
'''
bicycle
bridge
bus
car
chimney
crosswalk
hydrant
mountain
other
palm
traffic lights
'''

data_dir = "C:/Users/Liam/Desktop/Programming/Python/tensorEnv/Machine Learning/CaughtCha/Dataset"
path, dirs, files = next(os.walk(data_dir))
image_count = len(files)

# image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)

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

AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)  # Cache images into memory after first epoch
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)  # Prefetch overlaps preprocessing and model exec while training

normalization_layer = layers.experimental.preprocessing.Rescaling(
    1. / 255)  # RGB values aren't ideal for a neural network

num_classes = 11

data_augmentation = keras.Sequential(
    [
        layers.experimental.preprocessing.RandomFlip("horizontal", input_shape=(img_height,
                                                                                img_width,
                                                                                3)),
        layers.experimental.preprocessing.RandomRotation(0.1),
        layers.experimental.preprocessing.RandomZoom(0.1)
    ]
)

###
###  MODEL CODE WENT HERE
###

###
###   CAPTCHA STEALS CODE WENT HERE
###

# for _ in range(10):
correctAnswers = 0

model = Sequential([
    data_augmentation,
    layers.experimental.preprocessing.Rescaling(1. / 255),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.14),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
])
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
# model.summary()
# Summarize the model
# plot_model(model, 'model.png', show_shapes=True)

epochs = 22
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

# model = keras.models.load_model("CaptchaNetwork")
directory = r'C:\Users\Liam\Desktop\Programming\Python\tensorEnv\Machine Learning\CaughtCha\Captcha Steals'

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
    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
            .format(class_names[np.argmax(score)], 100 * np.max(score))
    )
    class_names_Array = np.array(class_names)[topTwoArgs.astype(int)]
    # print("Second Highest Confidence: ", class_names_Array)

    # className = str(class_names[np.argmax(score)])
