import numpy as np

'''
Descripción: Contiene forward y backward de las funciones de activación
'''

def sigmoid(Z):
    '''
    Implementa la función de activación sigmoide
    
    Arguments:
    Z -- Activación linear de la capa actual, de dimensiones(tamaño de la capa actual, número de ejemplos)
    
    Returns:
    A -- sigmoid(z), de dimensiones(tamaño de la capa actual, número de ejemplos)
    cache -- Z
    '''
    
    A = 1 / (1 + np.exp(-Z))
    cache = Z
    
    return A, cache

def relu(Z):
    '''
    Implementa la función de activación RELU
    
    Arguments:
    Z -- Activación linear de la capa actual, de dimensiones(tamaño de la capa actual, número de ejemplos)
    
    Returns:
    A -- RELU(z), de dimensiones(tamaño de la capa actual, número de ejemplos)
    cache -- Z
    '''
    
    A = np.maximum(0, Z)
    cache = Z

    return A, cache

def relu_backward(dA, cache):
    '''
    Implementa backpropagation para la función de activación RELU
    Arguments:
    dA -- Gradiente del costo respecto a la activación de la capa actual, de dimensiones iguales a A
    cache -- Z
    Returns:
    dZ -- Gradiente del costo respecto a Z
    '''

    Z = cache

    dZ = np.array(dA, copy = True)

    dZ[Z <= 0] = 0

    return dZ


def sigmoid_backward(dA, cache):
    '''
    Implementa backpropagation para la función de activación sigmoide
    Arguments:
    dA -- Gradiente del costo respecto a la activación de la capa actual, de dimensiones iguales a A
    cache -- Z
    Returns:
    dZ -- Gradiente del costo respecto a Z
    '''

    Z = cache

    s = 1 / (1 + np.exp(-Z))

    dZ = dA * s * (1 - s)

    return dZ