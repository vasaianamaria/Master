# 1. Here have 8 images, read the images, flatten them, then store them in a numpy array.
# Before storing the images divide them by 255.
# The numpy array should have the dimension equals to (8, 12288), in order to check the dimension use the shape attribute.

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# List of image file names
image_names = ['01.png', '02.png', '03.png', '04.png', '05.png', '06.png', '07.png', '08.png']

# Initialize an empty array to hold the image data with shape (8, 12288)
image_data = np.empty((len(image_names), 12288))  # Adapting for varying number of images

# Iterate through the image files using their names
for i, image_name in enumerate(image_names):
    # Open the image
    img = Image.open(f'images/{image_name}')

    # Ensure the image is of size (64, 64)
    img = img.resize((64, 64))

    # Convert the image to a NumPy array and normalizes pixel values to the range [0, 1]
    # by dividing by 255 (the maximum pixel value).
    img_array = np.array(img) / 255.0

    # Flattens the 3D image array (height, width, channels) into a 1D array and add it to the image_data array
    image_data[i, :] = img_array.flatten()

# Check the shape of the resulting array
print(image_data.shape)



#INFOS:
# An image size of 64x64 pixels means there are 64 pixels in width and 64 pixels in height, totaling 64 * 64 = 4096 pixels in the image.
# Each pixel has 3 values corresponding to the RGB channels, so you have 4096 pixels * 3 values/pixel = 12288 values
# to represent one image when it’s flattened into a 1D array

# 2.  Compute the Mean of the Images

mean_image = np.mean(image_data, axis=0) #will contain the average of all the 8 images at each pixel position, the mean image of all 8 images

# 3.
normalized_images = image_data - mean_image #broadcasting

# 4. Normalize the Images

w = np.load('coefs.npy')
b = np.load('bias.npy')


def softmax(normalized_images):
    e_z = np.exp(normalized_images - np.max(normalized_images, axis=1, keepdims=True))
    return e_z / np.sum(e_z, axis=1, keepdims=True)


# Make sure your images are flattened and normalized
# X_normalized = normalize_and_flatten(X)

# Compute y_hat
z = np.dot(normalized_images, w) + b
y_hat = softmax(z)

