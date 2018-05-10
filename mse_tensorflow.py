from __future__ import print_function

import numpy
import matplotlib.pyplot as plt
import pandas

from skmatrix import normalizer
from skmatrix import categorizer

import tensorflow as tf

numpy.random.seed(7)

input_file = 'full_entrada_salida_pesos_151.csv'

# COLUMNAS:
#  0     1   2    3     4         5      6         7          8      9
# index,dow,dom,month,year,in_demand,out_demand,prev_holy,pos_holy,minfl
vars_df = pandas.read_csv(input_file, usecols=[1, 2, 3, 7, 8])
demand_df = pandas.read_csv(input_file, usecols=[5, 6])

DIFF_DEMAND = demand_df.values[:, 0] - demand_df.values[:, 1]
DIFF_DEMAND.resize((len(DIFF_DEMAND), 1))

#  0     1   2    3     4         5      6         7          8      9
# index,dow,dom,month,year,in_demand,out_demand,prev_holy,pos_holy,minfl
dates_ds = pandas.read_csv(input_file, usecols=[2, 3, 4]).values

vars_ds = vars_df.values.astype('float64')
demand_ds = demand_df.values.astype('float64')

# reasigno a demand_ds como la demanda neta (entrada - salida)
in_demand = demand_ds[:, 0]
out_demand = demand_ds[:, 1]
demand_ds = in_demand - out_demand
demand_ds.resize((len(demand_ds), 1))

# convierto a la demanda en una categoria one_hot
categorized_demand, demand_bags = categorizer.categorize_real(demand_ds, 10)
one_hot_demand = categorizer.one_hot(categorized_demand)

# convierto a los dias de la semana en un arreglo one_hot
vars_w_categorized_dow = categorizer.categorize_int(vars_ds, 0)
one_hot_dow = categorizer.one_hot(vars_w_categorized_dow)
final_demand = one_hot_dow.copy()

# adjunto la demanda del dia anterior al arreglo de variables
vars_w_previous_demand = numpy.hstack( ( vars_ds[1:] , final_demand[:-1] ) )

# defino las variables categorizadas
final_vars = numpy.hstack((one_hot_dow[1:], vars_w_previous_demand[:, 1:]))
final_demand = final_demand[1:] # alineo la demanda final con las variables de entrada

# CONSTRUCCION DE LA RED -----------------------------------------------------------------------------------------------------------

# Parameters
learning_rate = 0.001
training_epochs = 300

# defino la cantidad de registros de entrenamiento y de prueba a partir del batch_size
# batch_size = 5
test_size = 30
# train_size = batch_size * int((len(vars_ds) - test_size) / batch_size)
train_size = len(final_vars) - test_size
batch_size = train_size

display_step = 1


train_in_ds = final_vars[0:train_size]  # dataset de entrnamiento de entrada
train_out_ds = final_demand[0:train_size]  # dataset de entrenamiento de salida o clases

test_in_ds = final_vars[train_size:]  # dataset de pruebas de entrada
test_out_ds = final_demand[train_size:]  # dataset de pruebas de salida

n_input = train_in_ds.shape[1]  # defino el tamano de la capa de entrada
n_classes = train_out_ds.shape[1]  # defino el tamano de la capa de salida / numero de clases a clasificar

# el tamano de las capas ocultas es igual al tamano de la capa de salida
n_hidden_1 = int((train_in_ds.shape[1] + train_out_ds.shape[1]) / 2)
n_hidden_2 = int((train_in_ds.shape[1] + train_out_ds.shape[1]) / 2)

def build_random_weight_tensor(in_size, out_size):
    """ Construye un tensor de in_size * out_size elementos con valores aleatorios siguiendo la distribucion normal.
    Cada valor representa el peso de la entrada 'i' hacia la neurona 'j' (siendo i una fila y j una columna) """
    return tf.random_normal(shape=[in_size, out_size])


def build_random_bias_tensor(out_size):
    """ Crea un tensor de umbrales de activacion para una capa de la red neuronal """
    return tf.random_normal(shape=[out_size])


# NOTAR QUE LOS PESOS Y BIASES SON VARIABLES
# Esto se debe a que las variables son los unicos componentes del modelo que pueden ser reasignados y por lo tanto, optimizados

# Store layers weight & bias
# creo tres tensores. Cada tensor representa los pesos de la entrada para cada neurona
# hidden_1 representa los pesos de las neuronas de entrada con la primera capa oculta (hidden_1)
# hidden_2 representa los pesos de las salidas de hidden_1 con la segunda capa
# hidden_3 representa los pesos de las salidas de hidden_2 con la salida
weights = {
    'hidden_1': tf.Variable(build_random_weight_tensor(n_input, n_hidden_1)),
    'hidden_2': tf.Variable(build_random_weight_tensor(n_hidden_1, n_hidden_2)),
    'out': tf.Variable(build_random_weight_tensor(n_hidden_2, n_classes))
}

# defino los umbrales de activacion
biases = {
    'bias_1': tf.Variable(build_random_bias_tensor(n_hidden_1)),
    'bias_2': tf.Variable(build_random_bias_tensor(n_hidden_2)),
    'out': tf.Variable(build_random_bias_tensor(n_classes))
}


# Create model
def multilayer_perceptron(x):
    # Hidden fully connected layer with 256 neurons
    # Cada capa resuelve: x * w + b
    layer_1 = tf.matmul(x, weights['hidden_1']) + biases['bias_1']
    
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.matmul(layer_1, weights['hidden_2']) + biases['bias_2']
    
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    
    return out_layer


# El placeholder X sera un tensor de dimensiones [? , n_input]
# El placeholder Y sera un tensor de dimensiones [? , n_classes]
# Los placeholders permiten inyectar valores al modelo
# Suelen usarse para inyectar los valores de entrada y salida para el entrenamiento
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_classes])

# Construct model
logits = multilayer_perceptron(X)

# Defino la funcion de perdida y el optimizador que debe minimizar dicha funcion
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Esta funcion define un step u operacion de tensorflow que inicializara variables (los tf.Variable)
init_op = tf.global_variables_initializer()


sess = tf.Session()

sess.run(init_op)

# Training cycle
for epoch in range(training_epochs):
    avg_cost = 0.0
    # Run optimization op (backprop) and cost op (to get loss value)
    _, cost = sess.run([train_op, loss_op], feed_dict={X: train_in_ds, Y: train_out_ds})
    # Compute average loss
    avg_cost += cost
    if (epoch % display_step == 0):
        print("Epoch:", '%04d' % (epoch + 1), "cost={:.9f}".format(avg_cost))

print("Optimization Finished!")

# Test model
pred = tf.nn.softmax(logits)  # Aplico funcion de activacion SOFTMAX a la capa final
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1)) # determino punto a punto si la prediccion es correcta

accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print("Accuracy:", accuracy.eval({X: test_in_ds, Y: test_out_ds} , session=sess))


sess.close()
