import cv2
import skimage
import numpy as np
import tensorflow as tf



def load_image(path):

    image = []
    img = tf.image.grayscale_to_rgb(tf.image.decode_image(tf.io.read_file(path)))
    img = tf.image.resize(img, [150, 150])
    img = np.asarray(img)

    image.append(img)

    return np.asarray(image)

def load_model():
    my_model = tf.keras.models.load_model('vggmodel')
    return my_model