import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math

# Convolutional Layer 1.
filter_size1 = 5  # Filtros son de 5 x 5 pixeles.
num_filters1 = 16  # Hay 16 de estos filtros.

# Convolutional Layer 2.
filter_size2 = 5  # Filtros son de 5x5 pixeles.
num_filters2 = 36  # Hay 16 de estos filtros.

# Convolutional Layer 3.
filter_size3 = 5  # Filtros son de 5x5 pixeles.
num_filters3 = 64  # Hay 16 de estos filtros.

# Fully-connected layer.
fc_size = 128  # Número de neuronas de la capa fully-connected

# Dropout probability
keep_prob = tf.placeholder(tf.float32)
prob = 1.0

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


def new_conv_layer(input,  # Capa anterior.
                   num_input_channels,  # Numero de canales de la capa anterior.
                   filter_size,  # Ancho y alto de cada filtro.
                   num_filters,  # Número de filtros.
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



def new_fc_layer(input,  # Capa anterior.
                 num_inputs,  # Numero de entradas.
                 num_outputs,  # Numero de salidas.
                 use_relu=True,  # Decide si usar ReLU o no.
                 use_dropout=True):  # Decide si usar dropout

    # Crear pesos y biases.
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    # Evaluar capa fully connected.
    layer = tf.matmul(input, weights) + biases

    # Usar ReLU?
    if use_relu:
        layer = tf.nn.relu(layer)

    if use_dropout:
        layer = tf.nn.dropout(layer, keep_prob)

    return layer


x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')

x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])

y_true = tf.placeholder(tf.float32, shape=[None, 10], name='y_true')

y_true_cls = tf.argmax(y_true, dimension=1)

layer_conv1, weights_conv1 = \
    new_conv_layer(input=x_image,
                   num_input_channels=num_channels,
                   filter_size=filter_size1,
                   num_filters=num_filters1,
                   use_pooling=True)

layer_conv2, weights_conv2 = \
    new_conv_layer(input=layer_conv1,
                   num_input_channels=num_filters1,
                   filter_size=filter_size2,
                   num_filters=num_filters2,
                   use_pooling=True)

layer_conv3, weights_conv3 = \
    new_conv_layer(input=layer_conv2,
                   num_input_channels=num_filters2,
                   filter_size=filter_size3,
                   num_filters=num_filters3,
                   use_pooling=True)

layer_flat, num_features = flatten_layer(layer_conv3)

layer_fc1 = new_fc_layer(input=layer_flat,
                         num_inputs=num_features,
                         num_outputs=fc_size,
                         use_relu=True,
                         use_dropout=True)

layer_fc2 = new_fc_layer(input=layer_fc1,
                         num_inputs=fc_size,
                         num_outputs=num_classes,
                         use_relu=False,
                         use_dropout=False)

y_pred = tf.nn.softmax(layer_fc2)
y_pred_cls = tf.argmax(y_pred, dimension=1)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2, labels=y_true)

cost = tf.reduce_mean(cross_entropy)

optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

session = tf.Session(config=tf.ConfigProto(log_device_placement=True))

session.run(tf.global_variables_initializer())

# Entrenamiento realizado por batches.
train_batch_size = 100

# Contador de iteraciones.
total_iterations = 0

# Progreso
iteracion = []
prog = []
ciclo = []
cic_prog = []


def optimize(num_iterations):
    global total_iterations, iteracion, prog, ciclo, cic_prog

    # Tiempo de inicio
    start_time = time.time()

    for i in range(total_iterations, total_iterations + num_iterations):

        # Obtener batch de conjunto de entrenamiento.
        x_batch, y_true_batch = data.train.next_batch(train_batch_size)

        # Se pone el batch en un diccionario asignándole nombres de las
        # variables placeholder antes definidas.
        feed_dict_train = {x: x_batch, y_true: y_true_batch, keep_prob: prob}

        # Ejecución del optimizador con los batches del diccionario.
        session.run(optimizer, feed_dict=feed_dict_train)

        # Se imprime elprogreso cada 100 iteraciones.
        if i % 50 == 0:
            acc = session.run(accuracy, feed_dict=feed_dict_train)
            iteracion.append(i)
            prog.append(acc)
            if i % int(55000 / train_batch_size) == 0:
                ciclo.append(i)
                cic_prog.append(acc)
            msg = "Iterations: {0:>6}, Training Accuracy: {1:>6.1%}"
            print(msg.format(i, acc))

    ciclo.append(i)
    cic_prog.append(acc)

    acc = session.run(accuracy, feed_dict=feed_dict_train)
    msg = "Iterations: {0:>6}, Training Accuracy: {1:>6.1%}"
    print(msg.format(i, acc))
    # Actualización del número de iteraciones.
    total_iterations += num_iterations

    # Tiempo de finalización.
    end_time = time.time()

    # Tiempo transcurrido.
    time_dif = end_time - start_time

    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))


# Dividir test set en batches. (Usa batches mas pequeños si la RAM falla).
test_batch_size = 256


def print_test_accuracy():
    # Número de imagenes en test-set.
    num_test = len(data.test.images)

    # Crea arreglo para guardar clases predichas.
    cls_pred = np.zeros(shape=num_test, dtype=np.int)

    # Calcular clases predichas.
    i = 0
    while i < num_test:
        j = min(i + test_batch_size, num_test)
        images = data.test.images[i:j, :]
        labels = data.test.labels[i:j, :]
        feed_dict = {x: images,
                     y_true: labels,
                     keep_prob: prob}

        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)
        i = j

    # Labels reales.
    cls_true = data.test.cls

    # Arreglo booleano de clasificaciones correctas.
    correct = (cls_true == cls_pred)

    # Matriz de confusion
    conf_mat = confusion_matrix(cls_true, cls_pred)

    # Número de clasificaciones correctas.
    correct_sum = correct.sum()

    # Accuracy
    acc = float(correct_sum) / num_test
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, correct_sum, num_test))
    print("Confusion matrix obtained:")
    print(conf_mat)

print("Testing with three convolution layers")
# Definir número de iteraciones que desea entrenar a la red
optimize(num_iterations=5500)

print_test_accuracy()

plt.plot(ciclo, cic_prog)
plt.title("Convergencia por época")
plt.ylabel("Porcentaje de precisión")
plt.xlabel("Número de iteraciones")
plt.show()


# Si usted ejecuta esta linea de código debe cerrar el notebook y reiniciarlo.
# Es solo para informar como liberar los recursos que ocupa TF.
# session.close()
