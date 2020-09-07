from .activations import relu, sigmoid, relu_backward, sigmoid_backward
import numpy as np

'''
Descripción:  Contiene un modelo de red neuronal y sus funciones respectivas
'''


def initialize_parameteres(dims):
    '''
    Inializa los pesos y sesgos de la red neuronal
    
    Arguments:
    dims -- python array (list) que contiene las dimensiones de cada capa de la red
    
    Returns:
    parameters -- python dictionary que contiene los parametros "W1",  "b1", ..., "Wn", "bn":
        W1 -- matriz de pesos, de dimensiones (dims[l], dims[l-1])
        b1 -- vector de sesgos, de dimensiones (dims[l], dims[1]) 
    '''

    parameters = {}

    ldims = len(dims)

    for l in range(1, ldims):
        parameters["W" + str(l)]: np.random.randn(dims[l], dims[l - 1]) * 0.01
        parameters["b" + str(l)]: np.zeros((dims[l], 1))

    return parameters


def linear_forward(A, W, b):
    '''
    Implementa la propagación linear hacia adelante
    
    Arguments:
    A -- activación de la capa anterior o input data, de dimensiones (tamaño de la capa anterior, número de ejemplos)
    W -- pesos de la capa actual, de dimensiones (tamaño de la capa actual, tamaño de la capa anterior)
    b -- sesgos de la capa actual, de dimensiones (tamaño de la capa actual, 1)
    
    Returns:
    Z -- input de la activación de la capa actual
    cache -- Python tuple que contiene "A", "W", "b"
    '''

    Z = np.dot(W, A) + b
    cache = (A, W, b)

    return Z, cache


def linear_activation_forward(A_prev, W, b, activation):
    '''
    Implementa la propagación hacia adelante Linear->Activación de una capa
    
    Arguments:
    A_prev -- Activación de la capa anterior, de dimensiones (tamaño de la capa anterior, número de ejemplos)
    W -- Pesos de la capa actual, de dimensiones (tamaño de la capa actual, tamaño de la capa anterior)
    b -- sesgos de la capa actual, de dimensiones (tamaño de la capa actual, 1)
    activation -- string con el nombre de la activación que será usada en esta capa: "Relu", "Sigmoid"
    
    Returns:
    A -- Activación de la capa actual, de dimensiones (tamaño de la capa actual, número de ejemplos)
    cache -- python tupla que contiene el cache linear y de la activación
    '''

    if activation == "Sigmoid":
        Z, linear_c = linear_forward(A_prev, W, b)
        A, activacion_c = sigmoid(Z)

    elif activation == "Relu":
        Z, linear_c = linear_forward(A_prev, W, b)
        A, activacion_c = relu(Z)

    cache = (linear_c, activacion_c)

    return A, cache


def model_forward(X, parameteres):
    '''
    Implementa un paso hacia adelante para el modelo:
        - Para las capas 1, ..., l - 1 implementa RELU
        - Para la capa l usa la función sigmoide
        
    Arguments:
    X -- input de la red neuronal, de dimensiones (input_size, número de ejemplos)
    parameteres -- python array (list) que contiene los parametros de la red neuronal
    
    Returns:
    Al -- activación de la l capa de la red neuronal
    caches -- lista de los caches de todas las capas, indexados de 0 a l - 1
    '''

    caches = []
    A = X
    L = len(parameteres) // 2

    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameteres["W" + str(l)], parameteres["b" + str(l)], "Relu")
        caches.append(cache)

    AL, cache = linear_activation_forward(A, parameteres["W" + str(L)], parameteres["b" + str(L)], "Sigmoid")
    caches.append(cache)

    return AL, caches


def cost_function(AL, Y):
    '''
    Implementa la función de costo para la red neuronal

    Arguments:
    AL -- Vector de probabilidades obtenido de la red neuronal, de dimensiones(1, número de ejemplos)
    Y -- True label vector, de dimensiones(1, número de ejemplos)

    Returns:
    cost -- Cross-entropy cost
    '''

    cost = -1 / AL.shape[1] * np.sum(Y * np.log(AL) + (1 - Y) * np.log(1 - AL))
    cost = np.squeeze(cost)

    return cost


def linear_backward(dZ, cache):
    '''
    Implementa la parte lineal del backpropagation de la capa actual
    Argumens:
    dZ -- Gradiente del costo respecto a la parte lineal
    cache -- python tuple que contiene A_prev, W, b obtenido de la propagación hacia adelante de la capa actual

    Returns:
    dA_prev -- Gradiente del costo respecto a la activación de la capa anterior, de dimensiones iguales a A_prev
    dW -- Gradiente del costo respecto a los pesos de la capa actual, de dimensiones iguales a W
    db -- Gradiente del csto respecto a los sesgos de la capa actual, de dimensiones iguales a b
    '''

    A_prev, W, b = cache

    m = A_prev.shape[1]

    dW = 1 / m * np.dot(dZ, A_prev.T)
    db = 1 / m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db


def linear_activation_forward(dA, cache, activation):
    '''
    Implementa backpropagation para la LINEAR->ACTIVATION de la capa actual
    Arguments:
    dA: Gradiente del costo respecto a la función de activación, de dimensiones igual a A
    cache: tupla que contiene el cache de la parte lineal y el cache de la activación
    activation: string que contiene el nombre de la activación usada, "Sigmoid" o "Relu"

    Returns:
    dA_prev -- Gradiente del costo respecto a la activación de la capa anterior, de dimensiones iguales a A_prev
    dW -- Gradiente del costo respecto a los pesos de la capa actual, de dimensiones iguales a W
    db -- Gradiente del costo respecto a los sesgos de la capa actual, de dimensiones iguales a b
    '''

    linear_c, activation_c = cache

    if activation == "Relu":
        dZ = relu_backward(dA, activation_c)
        dA_prev, dW, db = linear_backward(dZ, linear_c)

    elif activation == "Sigmoid":
        dZ = sigmoid_backward(dA, activation_c)
        dA_prev, dW, db = linear_backward(dZ, linear_c)

    return dA_prev, dW, db