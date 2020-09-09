from model import *
import matplotlib.pyplot as plt
import numpy as np

''''
Descripción: contiene la arquitectura de un modelo de red neuronal de L capas
'''


def L_layer_model(X, Y, dims, cost_type, Lactivation, learning_rate=0.009, num_iterations=3000, print_cost=False):
    '''
    Implementa un modelo de red neuronal de L capas, [Linear->RELU]*(L-1)->Linear->Lactivation

    Arguments:
    X -- data, arreglo numpy de dimensiones(138, número de ejemplos)
    Y -- True label, arreglo numpy de dimensiones(1, número de ejemplos)
    dims -- Topología de la red neuronal
    Lactivation -- String que contiene la activación de la L capa de la red neuronal, "Sigmoid" o  "Relu"
    learning_rate -- Parametro de aprendizaje de la red neuronal
    num_iterations -- Iteraciones de la fase de entrenamiento
    print_cost -- Variable que controla la impresión del costo cada 100 pasos

    Returns:
    parameteres -- parametros aprendidos por la red neuronal, usados para predecir
    '''
    costs = []
    parameters = initialize_parameters(dims)

    for i in range(num_iterations):
        AL, caches = model_forward(X, parameters, Lactivation)

        cost = cost_function(AL, Y, cost_type)

        grads = model_backward(AL, Y, caches, cost_type, Lactivation)

        parameters = update_parameters(parameters, grads, learning_rate)

        if print_cost and i % 100 == 0:
            print("Cost after iteration {}: {}".format(i, cost))
            costs.append(cost)

    plt.rcParams['figure.figsize'] = (5.0, 4.0)  # set default size of plots
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'

    #Plot the cost
    plt.ylabel("Cost")
    plt.xlabel("Iterations (per hundreds)")
    plt.title("Learning rate = " + str(learning_rate))
    plt.plot(np.squeeze(list(range(num_iterations // 100))),np.squeeze(costs))
    #plt.imshow()

    return parameters