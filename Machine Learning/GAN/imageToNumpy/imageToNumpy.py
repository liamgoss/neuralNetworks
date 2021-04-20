import os

from PIL import Image

import numpy as np

# Directory containing images you wish to convert
input_dir = "/Users/Liam/Desktop/Programming/Python/tensorEnv/Machine Learning/GAN/art_dataset"
directories = os.listdir(input_dir)
print(directories)
for filename in os.listdir(directory):
    location = "Captcha Steals/" + str(filename)

    im = Image.open(image)

    im = (np.array(im))

    r = im[:, :, 0].flatten()

    g = im[:, :, 1].flatten()

    b = im[:, :, 2].flatten()

    label = [1]

    out = np.array(list(label) + list(r) + list(g) + list(b), np.uint8)

    out.tofile("out.bin")
