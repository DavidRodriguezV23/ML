from activations import relu, sigmoid, relu_backward, sigmoid_backward
import numpy as np

'''
Descripción:  Contiene un las funciones de la red neuronal
'''


def initialize_parameters(dims):
    '''
    Inializa los pesos y sesgos de la red neuronal
    
    Arguments:
    dims -- python array (list) que contiene las dimensiones de cada capa de la red
    
    Returns:
    parameters -- python dictionary que contiene los parametros "W1",  "b1", ..., "Wn", "bn":
        W1 -- matriz de pesos, de dimensiones (dims[l], dims[l-1])
        b1 -- vector de sesgos, de dimensiones (dims[l], dims[1]) 
    '''

    # print(len(dims))

    parameters = {}

    ldims = len(dims)

    for l in range(1, ldims):
        parameters["W" + str(l)] = np.abs(np.random.randn(dims[l], dims[l - 1])) * np.sqrt(2 / dims[l - 1])
        parameters["b" + str(l)] = np.zeros((dims[l], 1))

    # print(parameters)

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


def model_forward(X, parameters, Ltype):
    '''
    Implementa un paso hacia adelante para el modelo:
        - Para las capas 1, ..., l - 1 implementa RELU
        - Para la capa l usa la función de tipo Ltype
        
    Arguments:
    X -- input de la red neuronal, de dimensiones (input_size, número de ejemplos)
    parameters -- python array (list) que contiene los parametros de la red neuronal
    Ltpye -- String que contiene el tipo de activación de la ultima capa de la red neuronal, "Sigmoid" o "Relu"
    
    Returns:
    Al -- activación de la l capa de la red neuronal
    caches -- lista de los caches de todas las capas, indexados de 0 a l - 1
    '''

    caches = []
    A = X
    L = len(parameters) // 2

    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters["W" + str(l)], parameters["b" + str(l)], "Relu")
        caches.append(cache)

    if Ltype == "Sigmoid":
        AL, cache = linear_activation_forward(A, parameters["W" + str(L)], parameters["b" + str(L)], "Sigmoid")
    elif Ltype == "Relu":
        AL, cache = linear_activation_forward(A, parameters["W" + str(L)], parameters["b" + str(L)], "Relu")

    caches.append(cache)

    return AL, caches


def cost_function(AL, Y, type):
    '''
    Implementa la función de costo para la red neuronal

    Arguments:
    AL -- Vector de probabilidades obtenido de la red neuronal, de dimensiones(1, número de ejemplos)
    Y -- True label vector, de dimensiones(1, número de ejemplos)
    type -- string que contiene el identificador de la función de costo, "Ce" para cross entropy, "Rmse" para root mean
    squared error, "Mse" para mean squared error

    Returns:
    cost -- retorna el costo
    '''

    if type == "Ce":
        cost = -1 / AL.shape[1] * np.sum(Y * np.log(AL) + (1 - Y) * np.log(1 - AL))
    elif type == "Rmse":
        cost = np.sqrt(1 / AL.shape[1] * np.sum(np.power(np.subtract(Y, AL), 2), axis=1))
    elif type == "Mse":
        cost = np.mean(np.power(np.subtract(Y, AL), 2))
    cost = np.squeeze(cost)

    return cost


def derivate_cost(AL, Y, type):
    '''
    Retorna el gradiente del costo asociado a la activación de la ultima capa
    Arguments:
    AL -- Valor predicho por la red neuronal, de dimensiones (1, número de ejemplos)
    Y -- True label del dataset, de dimensiones (1, número de ejemplos)
    type -- string que contiene el tipo de función usada para calcular el error, "Ce" para cross entropy, "Rmse" para
    root mean squared error, "Mse" para mean squared error

    Returns:
    dAL -- Gradiente del costo asociado a la activación de la ultima capa
    '''

    if type == "Ce":
        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    elif type == "Rmse":
        dAL = - 1 * np.divide(np.subtract(Y, AL), (np.sqrt((np.power(np.subtract(Y, AL), 2)))))
    elif type == "Mse":
        dAL = np.subtract(AL, Y)

    return dAL


def linear_backward(dZ, cache):
    '''
    Implementa la parte lineal del backpropagation de la capa actual
    Arguments:
    dZ -- Gradiente del costo respecto a la parte lineal
    cache -- python tuple que contiene A_prev, W, b obtenido de la propagación hacia adelante de la capa actual

    Returns:
    dA_prev -- Gradiente del costo respecto a la activación de la capa anterior, de dimensiones iguales a A_prev
    dW -- Gradiente del costo respecto a los pesos de la capa actual, de dimensiones iguales a W
    db -- Gradiente del costo respecto a los sesgos de la capa actual, de dimensiones iguales a b
    '''

    A_prev, W, b = cache

    m = A_prev.shape[1]

    dW = 1 / m * np.dot(dZ, A_prev.T)
    db = 1 / m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):
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


def model_backward(AL, Y, caches, costType, activationType):
    '''
    Implementa backpropagation para todo el modelo de red neuronal

    Arguments:
    AL --  Valor predicho por la red neuronal, de dimensiones (1, número de ejemplos)
    Y --  True label vector, de dimensiones (1, número de ejemplos)
    caches --  Lista de caches que contienen el cache lineal y de la activación:
        L - 1 RELU
        Para la capa L contiene activación dada al modelo
    costType -- String que contiene el tipo de costo usado para calcular el error de la red neuronal, "Ce" ó "Rmse"
    activationType -- String que contiene el tipo de activación usada en la última capa de la red neuronal, "Sigmoid" ó
    "Relu"

    Returns:
    grads -- Diccionario de gradientes dA, dW, db para cada una de las capas
    '''

    grads = {}
    L = len(caches)
    Y = Y.reshape(AL.shape)
    m = AL.shape[1]

    dAL = derivate_cost(AL, Y, costType)

    current_c = caches[L - 1]

    if activationType == "Sigmoid":
        grads["dA" + str(L - 1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL,
                                                                                                          current_c,
                                                                                                          "Sigmoid")
    elif activationType == "Relu":
        grads["dA" + str(L - 1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL,
                                                                                                          current_c,
                                                                                                          "Relu")

    for l in reversed(range(L - 1)):
        current_c = caches[l]
        grads["dA" + str(l)], grads["dW" + str(l + 1)], grads["db" + str(l + 1)] = linear_activation_backward(
            grads["dA" + str(l + 1)], current_c,
            "Relu")

    return grads


def update_parameters(parameters, grads, learning_rate):
    '''
    Aplica descenso del gradiente para actualizar los pesos y sesgos de la red neuronal
    Arguments:
    parameters -- python dictionary que contiene los parametros actuales de la red neuronal
    grads -- python dictionary que contiene los gradientes obtenidos de backpropagation
    learning_rate -- Learning rate del modelo

    Returns:
    parameters -- python dictionary que contiene los parametros actualizados de la red neuronal
    '''
    L = len(parameters) // 2

    #print("parameteres W1, b1: {}\n{}".format(parameters["W1"], parameters["b1"]))
    #print("\ngrads dW1, db1: {}\n{}".format(grads["dW1"], grads["db1"]))

    for l in range(L):
        parameters["W" + str(l + 1)] -= learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] -= learning_rate * grads["db" + str(l + 1)]

    return parameters
