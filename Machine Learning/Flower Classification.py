import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
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

#dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
#data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
#data_dir = pathlib.Path(data_dir)
data_dir = "D:/Downloads/flower_photos/flower_photos"
path, dirs, files = next(os.walk(data_dir))
image_count = len(files)

#image_count = len(list(data_dir.glob('*/*.jpg')))
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
    ax = plt.subplot(3, 3, i +1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")

# Configure the data set for performance
AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)  # Cache images into memory after first epoch
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)  # Prefetch pverlaps preprocessing and model exec while training

# Standardize the data
normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)  # RGB values aren't ideal for a neural network

# Create the model
# The model consists of three convolution blocks with a max pool layer in each of them.
# There's a fully connected layer with 128 units on top of it that is activated by a relu activation function.
# This model has not been tuned for high accuracy, the goal of this tutorial is to show a standard approach.

num_classes = 5
# Overfitting issue - not actually *learning*
# Fix this with data augmentation and Dropout

data_augmentation = keras.Sequential(
  [
    layers.experimental.preprocessing.RandomFlip("horizontal", input_shape=(img_height,
                                                                            img_width,
                                                                            3)),
    layers.experimental.preprocessing.RandomRotation(0.1),
    layers.experimental.preprocessing.RandomZoom(0.1)
  ]
)
'''
best = 0
for _ in range(5):

  # Dropout ~ New network from before
  model = Sequential([
    data_augmentation,
    layers.experimental.preprocessing.Rescaling(1./255),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
  ])
  model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
  model.summary()

  epochs = 20
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
  
  #plt.figure(figsize=(8, 8))
  #plt.subplot(1, 2, 1)
  #plt.plot(epochs_range, acc, label='Training Accuracy')
  #plt.plot(epochs_range, val_acc, label='Validation Accuracy')
  #plt.legend(loc='lower right')
  #plt.title('Training and Validation Accuracy')
  
  #plt.subplot(1, 2, 2)
  #plt.plot(epochs_range, loss, label='Training Loss')
  #plt.plot(epochs_range, val_loss, label='Validation Loss')
  #plt.legend(loc='upper right')
  #plt.title('Training and Validation Loss')
  #plt.show()
  
  # Predict on totally new data
  #sunflower_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg"
  #rose_url = "https://www.crossfityellowrose.com/wp-content/uploads/2019/10/photo-1552034602-da2ee2dd724b.jpeg"
  #os.path.join(_, hotdog.jpeg)
  #sunflower_path = tf.keras.utils.get_file('hotdog.jpeg')

  img = keras.preprocessing.image.load_img(
    "C:/Users/Liam/Desktop/Programming/Python/tensorEnv/Machine Learning/corndog.jpg", target_size=(img_height, img_width)
  )
  img_array = keras.preprocessing.image.img_to_array(img)
  img_array = tf.expand_dims(img_array, 0)  # Create a batch

  #pickle_in = open("studentmodel.pickle", "rb")
  #linear = pickle.load(pickle_in)

  predictions = model.predict(img_array)
  score = tf.nn.softmax(predictions[0])


  # daisy, dandelion, roses, sunflowers, tulips

  print("Predictions: ", predictions)
  print("\nArgmax: ", np.argmax(score))
  print("Sort: ", np.argsort(score))

  topTwoArgs = np.argpartition(score, -2)[-2:]

  print("\nArgpartition: ", topTwoArgs)
  print("Highest Confidence: ", class_names[np.argmax(score)])
  class_names_Array = np.array(class_names)[topTwoArgs.astype(int)]
  print("Second Highest Confidence: ", class_names_Array)


  print(
      "This image most likely belongs to {} with a {:.2f} percent confidence."
      .format(class_names[np.argmax(score)], 100 * np.max(score))
  )
  print("\n\nCurrent Score: ", 100 * np.max(score))
  print("Best: ", best)
  if 100 * np.max(score) > best and class_names[np.argmax(score)] == "tulips":
    best = 100 * np.max(score)
    print("\nNew Best: ", best)
    model.save("Flower Power")
  elif 100 * np.max(score) > best and class_names[np.argmax(score)] != "tulips":
    print("\nIncorrect Identification \nNot Saving")
'''

model = keras.models.load_model("Flower Power")

directory = r'C:\Users\Liam\Desktop\Programming\Python\tensorEnv\Machine Learning\Test Images'
for filename in os.listdir(directory):
  location = "Test Images/" + str(filename)
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
  print("Second Highest Confidence: ", class_names_Array)

# daisy, dandelion, roses, sunflowers, tulips

#print("Predictions: ", predictions)
#print("\nArgmax: ", np.argmax(score))
#print("Sort: ", np.argsort(score))



#print("\nArgpartition: ", topTwoArgs)
