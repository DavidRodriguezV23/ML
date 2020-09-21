import numpy as np
import PIL.Image
import tensorflow as tf
import matplotlib.pyplot as plt

'''
Descripción: Contiene las funciones para el proprocesado
'''


def tensor_to_image(tf_input):
    '''
    Función que convierte de tensor a imagen.
    :param tf_input: tensor.
    :return: img: imagen generada a partir del tensor.
    '''

    tf_input = tf_input * 255
    tf_input = np.array(tf_input, dtype=np.uint8)
    if np.ndim(tf_input) > 3:
        assert tf_input.shape[0] == 1
        tf_input = tf_input[0]
    img = PIL.Image.fromarray(tf_input)
    return img


def load_img(path):
    '''
    Carga una imagen a partir de su dirección.
    :param path: python string que contiene la dirección de la imagen.
    :return: img: imagen obtenida a partir de la dirección proporcionada.
    '''

    max_dim = 512
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]

    return img


def imshow(img, title=None):
    '''
    Muestra la imagen.
    :param img: imagen a mostrar.
    :param title: python string que contiene el titulo de la imagen.
    :return: None
    '''
    if len(img.shape) > 3:
        img = tf.squeeze(img, axis=0)
    plt.imshow(img)
    if title:
        plt.title(title)


def get_layers():
    '''
    Función que retorna los nombres de las capas a usar para la transferencia de estilo.
    :return: layers: python tuple que contiene la listas de capas usadas para el contenido, el estilo,
     los pesos de las capas de estilo y contenido.
    '''

    content_layers = ["block5_conv2"]
    style_layers = ["block1_conv1",
                    "block2_conv1",
                    "block3_conv1",
                    "block4_conv1",
                    "block5_conv1"]

    weight_layers_s = [1e-2,
                       1e-2,
                       1e-2,
                       1e-2,
                       1e-2]

    weight_layers_c = [1e4]

    layers = (content_layers, style_layers, weight_layers_c, weight_layers_s)

    return layers


def vgg_layers(layer_names):
    '''
    Retorna tf.keras.Model que contiene como outputs las capas que se encuentran en layer_names.
    :param layer_names: lista que contiene los nombres de las capas usadas para el contenido y el estilo.
    :return: model: tf.keras.Model que contiene las capas que se encuentran en layer_names.
    '''

    VGG = tf.keras.applications.VGG19(include_top=False, weights="imagenet")
    VGG.trainable = False
    outputs = [VGG.get_layer(name).output for name in layer_names]

    model = tf.keras.Model(inputs=[VGG.input], outputs=outputs)

    return model
