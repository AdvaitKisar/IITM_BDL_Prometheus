# Importing libraries
import tensorflow as tf
from PIL import Image
import numpy as np

def format_image(image:Image):
    '''
    Function: Formats the image to a format suitable for further preprocessing

    Input:-
    image [Image]: PIL Image file

    Output:-
    flat_image [np array]: Flattened image with 784 elements
    '''
    image = np.array(image) # Converting Image to NumPy Array
    image = image/255 # Normalizing the image
    
    # If image is an RGB image, it creates a grayscale image of same size
    if len(image.shape) == 3 and image.shape[-1] == 3:
        image = tf.image.rgb_to_grayscale(image)
    # If image is an RGBA image, it creates a grayscale image of same size
    if len(image.shape) == 3 and image.shape[-1] == 4:
        image = tf.reduce_mean(image, axis=2, keepdims=True)

    # Convert NumPy array to TensorFlow tensor
    image = tf.convert_to_tensor(image)
    
    # Resize the image to 28x28
    resized_image = tf.image.resize(image, [28, 28])
    # Convert tensor to NumPy array
    resized_image = resized_image.numpy()
    # Flatten the array to a 1d array of 784 elements
    flat_image = resized_image.flatten().reshape(1, -1)
    return flat_image # Flattened image is returned