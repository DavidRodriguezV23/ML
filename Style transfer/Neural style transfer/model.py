import tensorflow as tf
from utils import vgg_layers

'''
Descripción: Contiene las funciones para computar el costo y el modelo a usar.
'''


def gram_matrix(input_tensor):
    '''
    Genera gram matrix para el tensor dado.
    :param input_tensor: tensorflow tensor.
    :return: GA: gram matrix.
    '''

    result = tf.linalg.einsum("bijc,bijd->bcd", input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return result / num_locations

    '''m, n_H, n_W, n_C = input_tensor.get_shape().as_list()

    input_tensor = tf.transpose(tf.reshape(input_tensor, [n_H * n_W, n_C]), perm=[1, 0])

    GA = tf.matmul(input_tensor, tf.transpose(input_tensor))

    return GA'''


def content_cost(content_outputs, target_content_outputs, content_weights):
    '''
    Computa el costo asociado al contenido de la imagen entre el target y el generado.
    :param content_outputs: python dic que contiene el contenido generado.
    :param content_weighst: python dic que contiene los pesos "importancia" del contenido en el costo
    :param target_content_outputs: python list que contiene el output target.
    :return: loss: Costo asociado al contenido.
    '''

    loss = tf.add_n([tf.reduce_mean((content_outputs[name] - target_content_outputs[name]) ** 2) for name in
                     content_outputs.keys()])

    loss *= content_weights[0] / len(content_outputs.keys())

    return loss


def style_cost(style_outputs, target_style_outputs, style_weights):
    '''
    Computa el costo asociado al estilo entre el target y el generado.
    :param style_outputs: python dict que contiene el estilo generado.
    :param target_style_outputs: python dict que contiene el target para el estilo.
    :param style_weights: python list que contiene el peso para cada capa usada.
    :return: loss: python list que contiene los costos asociados al estilo.
    '''

    loss = tf.add_n(
        [tf.reduce_mean((style_outputs[name] - target_style_outputs[name]) ** 2) for name in
         style_outputs.keys()])

    loss *= style_weights[0] / len(style_outputs.keys())

    return loss


def style_content_loss(outputs, targets, weights, alpha=40, beta=10):
    '''
    Calcula el costo asociado al estilo y el contenido entre el target y el generado.
    :param outputs: outputs generados para la imagen a optimizar.
    :param targets: python tuple que contiene el target de contenido y el target de estilo.
    :param weights: python tuple que contiene los pesos asociados al costo.
    :param alpha: hiperparametro para la importancia del contenido en el costo.
    :param beta: hiperaparametro para la importancia del estilo en el costo.
    :return: loss: costo generado entre las dos imagenes.
    '''

    content_target, style_target = targets
    content_weights, style_weights = weights
    style_outputs = outputs["style"]
    content_outputs = outputs["content"]

    content_loss = content_cost(content_outputs, content_target, content_weights)
    style_loss = style_cost(style_outputs, style_target, style_weights)

    loss = alpha * content_loss + beta * style_loss

    return loss


def clip_0_1(img):
    '''
    Aplica tf.clip a la imagen.
    :param img: tf.Variable que contiene la imagen generada.
    :return: img procesada.
    '''
    return tf.clip_by_value(img, clip_value_min=0.0, clip_value_max=1.0)


@tf.function()
def train_step(model, opt, img, target, weights, alpha=40, beta=10, variation_weight=30):
    '''
    tf.function que genera un paso de forwardpropagation y backpropagation.
    :param model: SCModel.
    :param opt: tf.optimizer.
    :param img: tf.Variable que contiene la imagen generada.
    :param target: python tuple que contiene el target de estilo y contenido.
    :param weights: python tuple que contiene los pesos para las capas del contenido y el estilo.
    :param alpha: hiperparametro usado en el calculo del costo.
    :param beta: hiperparametro usado en el calculo del costo.
    :param variation_weight: hiperparametro para ajustar la variación de la imagen.
    :return: None
    '''

    with tf.GradientTape() as tape:
        outputs = model(img)
        loss = style_content_loss(outputs, target, weights, alpha, beta)
        loss += variation_weight * tf.image.total_variation(img)

    grad = tape.gradient(loss, img)
    opt.apply_gradients([(grad, img)])
    img.assign(clip_0_1(img))


class SCModel(tf.keras.models.Model):
    '''
    tf.keras.models.Model que genera el contenido y el estilo de la imagen dada.
    '''

    def __init__(self, style_layers, content_layers):
        '''
        constructor de SCModel.
        :param style_layers: python list que contiene el nombre de las capas a usar para el estilo.
        :param content_layers: python listque contiene el nombre de las capas a usar para el contenido.
        '''
        super(SCModel, self).__init__()
        self.vgg = vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.num_content_layers = len(content_layers)
        self.vgg.trainable = False

    def call(self, inputs):
        '''
        general el contenido y el estilo de la imagen dada.
        :param inputs: imagen a ingresar al modelo.
        :return: SCDict: diccionario que contiene el estilo y contenido de la imagen.
        '''
        inputs = inputs * 255.0
        preproccesed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preproccesed_input)
        style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                          outputs[self.num_style_layers:])
        style_outputs = [gram_matrix(style_output) for style_output in style_outputs]
        content_dict = {content_name: value for content_name, value in zip(self.content_layers, content_outputs)}
        style_dict = {style_name: value for style_name, value in zip(self.style_layers, style_outputs)}

        SCDict = {"content": content_dict, "style": style_dict}

        return SCDict
