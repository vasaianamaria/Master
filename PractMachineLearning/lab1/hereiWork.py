
#Matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Compute the x and y coordinates for points on a sine curve
x = np.arange(0, 3 * np.pi, 0.1) #np.arange([start, ]stop, [step, ])
y = np.sin(x)
print(x)
print(y)
# Plot the points using matplotlib
plt.plot(x, y)
plt.show()  # You must call plt.show() to make graphics appear.

#---------------------------------------------------------------------------------------

# To plot the points independently without interpolating the values, we have to set the third parameters of function ‘plot’:
plt.plot(x, y, 'o') # 'o' tells Matplotlib to use circles as markers for each point in the plot
plt.show()  # You must call plt.show() to make graphics appear.

#---------------------------------------------------------------------------------------

# Compute the x and y coordinates for points on sine and cosine curves
x = np.arange(0, 3 * np.pi, 0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)

# Plot the points using matplotlib
plt.plot(x, y_sin)
plt.plot(x, y_cos)
plt.xlabel('x axis label')
plt.ylabel('y axis label')
plt.title('Sine and Cosine')
plt.legend(['Sine', 'Cosine'])
plt.show()

#---------------------------------------------------------------------------------------

# Compute the x and y coordinates for points on sine and cosine curves
x = np.arange(0, 3 * np.pi, 0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)

# Set up a subplot grid that has height 2 and width 1,
# and set the first such subplot as active.
plt.subplot(2, 1, 1)

# Make the first plot
plt.plot(x, y_sin)
plt.title('Sine')

# Set the second subplot as active, and make the second plot.
plt.subplot(2, 1, 2)
plt.plot(x, y_cos)
plt.title('Cosine')

# Show the figure.
plt.show()

#---------------------------------------------------------------------------------------

import numpy as np
#from scipy.misc import imresize
from PIL import Image #(Python Imaging Library): Used for opening, manipulating, and saving image files.
import matplotlib.pyplot as plt

#Load and Convert Image to NumPy Array, , enabling numerical operations on the image data.
img = Image.open('cat.png')
img = np.array(img)

img_tinted = img * [1, 0.95, 0.9] #red remains the same, green is reduced to 95%, and blue to 90%.

# Show the original image
plt.subplot(1, 2, 1)
plt.imshow(img)

# Show the tinted image
plt.subplot(1, 2, 2)

# A slight gotcha with imshow is that it might give strange results
# if presented with data that is not uint8. To work around this, we
# explicitly cast the image to uint8 before displaying it.
plt.imshow(np.uint8(img_tinted)) #uint8 (unsigned 8-bit integer), required for plt.imshow() to display the image correctly
plt.show()
# from scipy.misc import imresize
# ImportError: cannot import name 'imresize' from 'scipy.misc' (/Library/Python/3.9/site-packages/scipy/misc/__init__.py)
#EXPLICATION: the imresize function has been removed from scipy.misc in recent versions of SciPy.
