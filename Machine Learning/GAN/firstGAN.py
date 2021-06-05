# example of a dcgan on cifar10
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


ImageFile.LOAD_TRUNCATED_IMAGES = True


# define the standalone discriminator model
def define_discriminator(in_shape=(32, 32, 3)):
    model = Sequential()
    # normal
    model.add(Conv2D(64, (3, 3), padding='same', input_shape=in_shape))
    model.add(LeakyReLU(alpha=0.2))
    # downsample
    model.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # downsample
    model.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # downsample
    model.add(Conv2D(256, (3, 3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # classifier
    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model


# define the standalone generator model
def define_generator(latent_dim):
    model = Sequential()
    # foundation for 4x4 image
    n_nodes = 256 * 4 * 4

    model.add(Dense(n_nodes, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((4, 4, 256)))
    # upsample to 8x8
    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # upsample to 16x16
    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # upsample to 32x32
    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # output layer
    model.add(Conv2D(3, (3, 3), activation='tanh', padding='same'))
    return model


# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model):
    # make weights in the discriminator not trainable
    d_model.trainable = False
    # connect them
    model = Sequential()
    # add generator
    model.add(g_model)
    # add the discriminator
    model.add(d_model)
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model


# load and prepare cifar10 training images
def load_real_samples():
    # load cifar10 dataset
    (trainX, _), (_, _) = load_data()
    # convert from unsigned ints to floats
    X = trainX.astype('float32')
    # scale from [0,255] to [-1,1]
    X = (X - 127.5) / 127.5
    return X


def make_square(img):
    cols, rows = img.size
    if rows > cols:
        pad = (rows - cols) / 2
        img = img.crop((pad, 0, cols, cols))
    else:
        pad = (cols - rows) / 2
        img = img.crop((0, pad, rows, rows))
    return img


def load_art_samples():
    filenames = []
    # Load custom abstract art dataset from director
    directory = r"C:\Users\Liam\Desktop\Programming\Python\tensorEnv\Machine Learning\GAN\art_dataset"
    batch_size = 32
    img_height = 32
    img_width = 32
    if os.path.exists('training_data.ob'):
        with open('training_data.ob', 'rb') as fp:
            training_data = pickle.load(fp)
    else:
        training_data = []

        folderNames = []
        num = 0
        for folder in os.listdir(directory):
            dir = directory + "\\" + folder

            for sample in os.listdir(dir):

                sampleLocation = dir + "\\" + sample
                #print(sampleLocation)
                img = Image.open(sampleLocation)
                img.convert('RGB')
                img.load()
                img = make_square(img)
                img = img.resize((img_width, img_height), Image.ANTIALIAS)
                #img = make_RGB(np.asarray(img))
                #img = make_RGB(img)

                if np.asarray(img).shape != (32,32,3):
                    print(img, " not within constraints, deleting...")
                    #os.remove(sampleLocation)
                    # potential bad one at 'C:\\Users\\Liam\\Desktop\\Programming\\Python\\tensorEnv\\Machine Learning\\GAN\\art_dataset\\class_b\\00a7b22ed63b9bde2255a21d67658436.jpg\
                else:
                    num = num + 1
                    print("Sample #", num)
                    training_data.append(np.asarray(img))

                    filenames.append(sampleLocation)
                    folderNames.append(folder)
        with open('training_data.ob', 'wb') as fp:
            pickle.dump(training_data, fp)




    #np.array(repaired_data[:])

    # http://chrisschell.de/2018/02/01/how-to-efficiently-deal-with-huge-Numpy-arrays.html
    # https://stackoverflow.com/questions/20928136/input-and-output-numpy-arrays-to-h5py


    # https://ipython-books.github.io/48-processing-large-numpy-arrays-with-memory-mapping/#:~:text=Sometimes%2C%20we%20need%20to%20deal,as%20a%20regular%20NumPy%20array.
    # Use "memory mapping"
    # Or, if that doesn't work, somehow train our data in chunks
# https://cs230.stanford.edu/blog/datapipeline/https://cs230.stanford.edu/blog/datapipeline/https://cs230.stanford.edu/blog/datapipeline/
 # look up queue in tf
    #hdf5_store = h5py.File("cache.hdf5", "a")
    print("Before reshape: ", training_data[0].shape)

    training_data = np.reshape(training_data, (-1, img_width, img_height, 3))
    #hdf5_store.create_dataset("trainedDataArray", (18000, 480, 480, 3), compression="gzip", data=training_data)
    print("After reshape: ", training_data[0].shape)

    training_data = training_data / 127.5 - 1
    print("After division, before saving")
    np.save("artTraining", training_data)  # Saves as "artTraining.npy"
    print("After saving")
    #training_dataFromFile = hdf5_store["trainedDataArray"][:]
    #training_dataFromFile = training_dataFromFile / 127.5 - 1
    #hdf5_store.close()

    #return training_dataFromFile
    return training_data

# select real samples
def generate_real_samples(dataset, n_samples):
    # choose random instances
    ix = randint(0, dataset.shape[0], n_samples)
    # retrieve selected images
    X = dataset[ix]
    # generate 'real' class labels (1)
    y = ones((n_samples, 1))
    return X, y


# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
    # generate points in the latent space
    x_input = randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input


# use the generator to generate n fake examples, with class labels
def generate_fake_samples(g_model, latent_dim, n_samples):
    # generate points in latent space
    x_input = generate_latent_points(latent_dim, n_samples)
    # predict outputs
    X = g_model.predict(x_input)
    # create 'fake' class labels (0)
    y = zeros((n_samples, 1))
    return X, y


# create and save a plot of generated images
def save_plot(examples, epoch, n=7):
    # scale from [-1,1] to [0,1]
    examples = (examples + 1) / 2.0
    # plot images
    for i in range(n * n):
        # define subplot
        pyplot.subplot(n, n, 1 + i)
        # turn off axis
        pyplot.axis('off')
        # plot raw pixel data
        pyplot.imshow(examples[i])
    # save plot to file

    filename = r"C:\Users\Liam\Desktop\Programming\Python\tensorEnv\Machine Learning\GAN\generated_images\generated_plot_e%03d.png" % (
                epoch + 1)
    pyplot.savefig(filename)
    pyplot.close()


# evaluate the discriminator, plot generated images, save generator model
def summarize_performance(epoch, g_model, d_model, dataset, latent_dim, n_samples=150):
    # prepare real samples
    X_real, y_real = generate_real_samples(dataset, n_samples)
    # evaluate discriminator on real examples
    _, acc_real = d_model.evaluate(X_real, y_real, verbose=0)
    # prepare fake examples
    x_fake, y_fake = generate_fake_samples(g_model, latent_dim, n_samples)
    # evaluate discriminator on fake examples
    _, acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)
    # summarize discriminator performance
    print('>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real * 100, acc_fake * 100))
    # save plot
    save_plot(x_fake, epoch)
    # save the generator model tile file
    filename = r"C:\Users\Liam\Desktop\Programming\Python\tensorEnv\Machine Learning\GAN\saved_models\generator_model_%03d.h5" % (
                epoch + 1)
    g_model.save(filename)



# train the generator and discriminator
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=50, n_batch=128):
    bat_per_epo = int(dataset.shape[0] / n_batch)
    #print("dataset.shape[0] = ", dataset.shape[0])
    #bat_per_epo = int(15000/n_batch)
    #bat_per_epo = int(bat_per_epo / n_batch)


    half_batch = int(n_batch / 2)
    # manually enumerate epochs
    for i in range(n_epochs):
        # enumerate batches over the training set
        for j in range(bat_per_epo):
            # get randomly selected 'real' samples
            X_real, y_real = generate_real_samples(dataset, half_batch)
            # update discriminator model weights
            d_loss1, _ = d_model.train_on_batch(X_real, y_real)
            # generate 'fake' examples
            X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
            # update discriminator model weights
            d_loss2, _ = d_model.train_on_batch(X_fake, y_fake)
            # prepare points in latent space as input for the generator
            X_gan = generate_latent_points(latent_dim, n_batch)
            # create inverted labels for the fake samples
            y_gan = ones((n_batch, 1))
            # update the generator via the discriminator's error
            g_loss = gan_model.train_on_batch(X_gan, y_gan)
            # summarize loss on this batch
            print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f' %
                  (i + 1, j + 1, bat_per_epo, d_loss1, d_loss2, g_loss))
        # evaluate the model performance, sometimes
        if (i + 1) % 10 == 0:
            summarize_performance(i, g_model, d_model, dataset, latent_dim)




# This involves first loading the model from file, then using it to generate images. The generation of each image requires a point in the latent space as input.
'''
Quick question on this, can the above code be used on images of higher resolution as well by simply changing the input 
and output dimensions from 32x32x3 to a higher value and changing how you upsampled/downsampled things, 
or basically are there any other tricks involved.

Yes, exactly.

Although you may need more fancy tricks to make the model stable:
https://machinelearningmastery.com/how-to-code-generative-adversarial-network-hacks/

OR TRY THIS

Yes, you can load the images and use them to train a gan, I show how to load images here:
https://machinelearningmastery.com/how-to-load-and-manipulate-images-for-deep-learning-in-python-with-pil-pillow/



No, we don’t train the model on random noise, we use random noise as input to synthesize new images.

You can learn more about how GANs work in general here:
https://machinelearningmastery.com/start-here/#gans


'''



# size of the latent space
latent_dim = 100
# create the discriminator
d_model = define_discriminator()
# create the generator
g_model = define_generator(latent_dim)
# create the gan
gan_model = define_gan(g_model, d_model)
# load image data
# dataset = load_real_samples()
dataset = load_art_samples()
print("TYPE" + str(type(dataset)))
# train model
train(g_model, d_model, gan_model, dataset, latent_dim)

'''
anim_file = 'genImages.gif'
with imageio.get_writer(anim_file, mode='I') as writer:
    filenames = glob.glob('generated_plote*.png')
    filenames = sorted(filenames)
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)
    image = imageio.imread(filename)
    writer.append_data(image)
'''
