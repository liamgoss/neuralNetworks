from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy import vstack
import numpy as np
from numpy.random import randn
from numpy.random import randint
from keras.datasets.cifar10 import load_data
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Dropout
from matplotlib import pyplot
import tensorflow as tf
from PIL import Image, ImageFile
import pandas as pd
import os
import pickle
import imageio
import glob
import h5py
import random
import time

start_time = time.time()
ImageFile.LOAD_TRUNCATED_IMAGES = True
def get_input(path):
    img = Image.open(path)
    return img

def get_output(path, label_file = None):
    img_id = path.split('\\')[-2].split('_')[-1]  # either "a" or "b"
    return img_id

def preprocess_input(img):
    def make_square(img):
        cols, rows = img.size
        if rows > cols:
            pad = (rows - cols) / 2
            img = img.crop((pad, 0, cols, cols))
        else:
            pad = (cols - rows) / 2
            img = img.crop((0, pad, rows, rows))
        return img

    img_height = 420
    img_width = 420
    img.convert('RGB')
    img.load()
    img = make_square(img)
    img = img.resize((img_width, img_height), Image.ANTIALIAS)

    if np.asarray(img).shape == (420, 420, 3):
        img = np.asarray(img) / 127.5 - 1.0
        return img

def image_generator(files, label_file, batch_size = 64):
    while True:
        # Select files for the batch
        batch_paths = np.random.choice(a=files, size=batch_size)
        batch_input = []
        batch_output = []

        for input_path in batch_paths:
            input = get_input(input_path)
            output = get_output(input_path, label_file)

            input = preprocess_input(input_path)
            batch_input += [input]
            batch_output += [output]

        batch_x = np.array(batch_input)
        batch_y = np.array(batch_output)

        yield (batch_x, batch_y)



n = 17769
s = int(n / 2)
filename = 'filePaths.csv'
skip = sorted(random.sample(range(n),n-s))
df = pd.read_csv(filename, skiprows=skip)
dataset = df.values.tolist()

print("--- %s seconds ---" % (time.time() - start_time))

'''
if os.path.exists('nparr.txt'):
    print("File exists...")
    with open('nparr.txt', 'rb') as fp:
        training_data = pickle.load(fp)
else:
    print("File does not exist...")
    image_arrays = list()
    directory = r"C:\\Users\Liam\Desktop\Programming\Python\\tensorEnv\Machine Learning\GAN\\art_dataset"

    def make_square(img):
        cols, rows = img.size
        if rows > cols:
            pad = (rows - cols) / 2
            img = img.crop((pad, 0, cols, cols))
        else:
            pad = (cols - rows) / 2
            img = img.crop((0, pad, rows, rows))
        return img

    img_height = 420
    img_width = 420
    num = 0
    for folder in os.listdir(directory):
        dir = directory + "\\" + folder

        for sample in os.listdir(dir):

            sampleLocation = dir + "\\" + sample
            # print(sampleLocation)
            img = Image.open(sampleLocation)
            #img.convert('RGB')
            img.load()
            img = make_square(img)
            img = img.resize((img_width, img_height), Image.ANTIALIAS)
            # img = make_RGB(np.asarray(img))
            # img = make_RGB(img)

            if np.asarray(img).shape != (420, 420, 3):
                print(img, " not within constraints, deleting...")
                # os.remove(sampleLocation)
                # potential bad one at 'C:\\Users\\Liam\\Desktop\\Programming\\Python\\tensorEnv\\Machine Learning\\GAN\\art_dataset\\class_b\\00a7b22ed63b9bde2255a21d67658436.jpg\
            else:
                num = num + 1
                print("Sample #", num)
                img = np.asarray(img) / 127.5 - 1.0
                image_arrays.append(np.asarray(img))
                del img



    with open("nparr.txt", 'wb') as fp:
        pickle.dump(image_arrays, fp)
    quit()

print("Before reshape: ", training_data[0].shape)

training_data = np.reshape(training_data, (-1, 420, 420, 3))
print("After reshape: ", training_data[0].shape)

#training_data = training_data / 127.5 - 1
#training_data = training_data * 1.0/127.5 - 1.0

print("After division, before saving")
np.save("artTraining", training_data)  # Saves as "artTraining.npy"
print("After saving")


'''

'''
dataset = tf.data.Dataset.list_files("C:/Users/Liam/Desktop/Programming/Python/tensorEnv/Machine Learning/GAN/classless_dataset/*.jpg")


DATASET_PATH = "C:/Users/Liam/Desktop/Programming/Python/tensorEnv/Machine Learning/GAN/art_dataset"
N_CLASSES = 2
IMG_HEIGHT = 64
IMG_WIDTH = 64
CHANNELS = 3

def read_images(dataset_path, batch_size):
    imagepaths, labels = list(), list()
    label = 0
    classes = sorted(os.walk(dataset_path).__next__()[1])
    for c in classes:
        c_dir = os.path.join(dataset_path, c)
        walk = os.walk(c_dir).__next__()
    for sample in walk[2]:
        if sample.endswith('.jpg') or sample.endswith('.jpeg'):
            imagepaths.append(os.path.join(c_dir, sample))
            labels.append(label)
    label += 1

    # Convert to Tensor
    imagepaths = tf.convert_to_tensor(imagepaths, dtype=tf.string)
    labels = tf.convert_to_tensor(labels, dtype=tf.int420)
    # Build a TF Queue, shuffle data
    image, label = tf.data.Dataset.from_tensor_slices([imagepaths, labels])#.shuffle(tf.shape(input_tensor, out_type=tf.int64)[0]).repeat(num_epochs)
    #image, label = tf.train.slice_input_producer([imagepaths, labels], shuffle=True)

    # Read images from disk
    image = tf.read_file(image)
    image = tf.image.decode_jpeg(image, channels=CHANNELS)

    image = tf.image.resize_images(image, [IMG_HEIGHT, IMG_WIDTH])

    # Normalize
    image = image * 1.0/127.5 - 1.0

    # Create batches
    X, Y = tf.train.batch([image, label], batch_size=batch_size, capacity=batch_size * 8, num_threads = 4)

    return X, Y

# Parameters
learning_rate = 0.001
num_steps = 10000
batch_size = 128
display_step = 100

# Network Parameters
dropout = 0.75 # Dropout, probability to keep units

# Build the data input
X, Y = read_images(DATASET_PATH, batch_size)


# Create model
def conv_net(x, n_classes, dropout, reuse, is_training):
    # Define a scope for reusing the variables
    with tf.variable_scope('ConvNet', reuse=reuse):

        # Convolution Layer with 420 filters and a kernel size of 5
        conv1 = tf.layers.conv2d(x, 420, 5, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv1 = tf.layers.max_pooling2d(conv1, 2, 2)

        # Convolution Layer with 420 filters and a kernel size of 5
        conv2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv2 = tf.layers.max_pooling2d(conv2, 2, 2)

        # Flatten the data to a 1-D vector for the fully connected layer
        fc1 = tf.contrib.layers.flatten(conv2)

        # Fully connected layer (in contrib folder for now)
        fc1 = tf.layers.dense(fc1, 1024)
        # Apply Dropout (if is_training is False, dropout is not applied)
        fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)

        # Output layer, class prediction
        out = tf.layers.dense(fc1, n_classes)
        # Because 'softmax_cross_entropy_with_logits' already apply softmax,
        # we only apply softmax to testing network
        out = tf.nn.softmax(out) if not is_training else out

    return out

# Because Dropout have different behavior at training and prediction time, we
# need to create 2 distinct computation graphs that share the same weights.

# Create a graph for training
logits_train = conv_net(X, N_CLASSES, dropout, reuse=False, is_training=True)
# Create another graph for testing that reuse the same weights
logits_test = conv_net(X, N_CLASSES, dropout, reuse=True, is_training=False)

# Define loss and optimizer (with train logits, for dropout to take effect)
loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=logits_train, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(logits_test, 1), tf.cast(Y, tf.int64))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float420))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Saver object
saver = tf.train.Saver()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    # Start the data queue
    tf.train.start_queue_runners()

    # Training cycle
    for step in range(1, num_steps+1):

        if step % display_step == 0:
            # Run optimization and calculate batch loss and accuracy
            _, loss, acc = sess.run([train_op, loss_op, accuracy])
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))
        else:
            # Only run the optimization op (backprop)
            sess.run(train_op)

    print("Optimization Finished!")

    # Save your model
    saver.save(sess, 'my_tf_model')

'''


