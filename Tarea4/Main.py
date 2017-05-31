
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math

# Convolutional Layer 1.
filter_size1 = 5          # Filtros son de 5 x 5 pixeles.
num_filters1 = 16         # Hay 16 de estos filtros.

# Convolutional Layer 2.
filter_size2 = 5          # Filtros son de 5x5 pixeles.
num_filters2 = 36         # Hay 16 de estos filtros.

# Fully-connected layer.
fc_size = 128             # Número de neuronas de la capa fully-connected

from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('data/MNIST/', one_hot=True)

print("Tamaños de los subconjuntos de la base de datos:")
print("Training-set:\t\t{}".format(len(data.train.labels)))
print("Test-set:\t\t{}".format(len(data.test.labels)))
print("Validation-set:\t\t{}".format(len(data.validation.labels)))

data.test.cls = np.argmax(data.test.labels, axis=1)
data.validation.cls = np.argmax(data.test.labels, axis=1)

# Las imágenes de MNIST son de 28 x 28 pixeles.
img_size = 28
# Tamaño de arreglos unidimensionales que podrían guardar los datos de estas imágenes.
img_size_flat = img_size * img_size
# Tupla que sirve para redimensionar arreglos.
img_shape = (img_size, img_size)
# Número de canales de color de las imágenes. Si las imágenes fueran a color, este número sería 3.
num_channels = 1
# Número de clases.
num_classes = 10

def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))

def new_conv_layer(input,              # Capa anterior.
                   num_input_channels, # Numero de canales de la capa anterior.
                   filter_size,        # Ancho y alto de cada filtro.
                   num_filters,        # Número de filtros.
                   use_pooling=True):  # Usar 2x2 max-pooling.

    # Forma de los filtros convolucionales (de acuerdo a la API de TF).
    shape = [filter_size, filter_size, num_input_channels, num_filters]

    # Creación de los filtros.
    weights = new_weights(shape=shape)

    # Creación de biases, uno por filtro.
    biases = new_biases(length=num_filters)

    # Creación de la operación de convolución para TensorFlow.
    # Notar que se han configurado los strides en 1 para todas las dimensiones.
    # El primero y último stride siempre deben ser uno.
    # Si strides=[1, 2, 2, 1], entonces el filtro es movido
    # de 2 en 2 pixeles a lo largo de los ejes x e y de la imagen.
    # padding='SAME' significa que la imagen de entrada se rellena
    # con ceros para que el tamaño de la salida se mantenga.
    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')

    # Agregar los biases a los resultados de la convolución.
    layer += biases

    # Usar pooling para hacer down-sample de la entrada.
    if use_pooling:
        # Este es 2x2 max pooling, lo que significa que se considera
        # una ventana de 2x2 y se selecciona el valor mayor
        # de los 4 pixeles seleccionados. ksize representa las dimensiones de
        # la ventana de pooling y el stride define cómo la ventana se mueve por la imagen.
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')

    # Rectified Linear Unit (ReLU).
    layer = tf.nn.relu(layer)

    # La función retorna el resultado de la capa y los pesos aprendidos.
    return layer, weights


def flatten_layer(layer):
    # Obtener dimensiones de la entrada.
    layer_shape = layer.get_shape()

    # Obtener numero de características.
    num_features = layer_shape[1:4].num_elements()

    # Redimensionar la salida a [num_images, num_features].
    layer_flat = tf.reshape(layer, [-1, num_features])

    # Las dimensiones de la salida son ahora:
    # [num_images, img_height * img_width * num_channels]
    # Retornar
    return layer_flat, num_features

def new_fc_layer(input,          # Capa anterior.
                 num_inputs,     # Numero de entradas.
                 num_outputs,    # Numero de salidas.
                 use_relu=True): # Decide si usar ReLU o no.

    # Crear pesos y biases.
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    # Evaluar capa fully connected.
    layer = tf.matmul(input, weights) + biases

    # Usar ReLU?
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer

x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')

x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])

y_true = tf.placeholder(tf.float32, shape=[None, 10], name='y_true')

layer_conv1, weights_conv1 = \
    new_conv_layer(input=x_image,
                   num_input_channels=num_channels,
                   filter_size=filter_size1,
                   num_filters=num_filters1,
                   use_pooling=True)