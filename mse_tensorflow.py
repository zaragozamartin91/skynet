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

RAW_IN_COL_COUNT = 5  # CANTIDAD DE COLUMNAS DE ENTRADA SIN CONTAR LAS PREDICCIONES DE DIAS ANTERIORES
DOW_COL, DOM_COL, MONTH_COL, PREVH_COL, POSTH_COL = range(RAW_IN_COL_COUNT)

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

# obtengo los valores de demanda categorizados
CAT_COUNT = 10
categorized_demand, demand_bags = categorizer.categorize_real(demand_ds, cat_count=CAT_COUNT)
one_hot_demand = categorizer.one_hot(categorized_demand)
final_demand = one_hot_demand.copy()

# convierto a los dias de la semana en un arreglo one_hot
vars_w_categorized_dow = categorizer.categorize_int(vars_ds, cat_col=DOW_COL)
one_hot_dow = categorizer.one_hot(vars_w_categorized_dow, cat_col=DOW_COL)

# convierto los dias del mes en un arreglo one_hot
vars_w_categorized_dom = categorizer.categorize_int(vars_ds, cat_col=DOM_COL)
one_hot_dom = categorizer.one_hot(vars_w_categorized_dom, cat_col=DOM_COL)

# convierto los meses en un arreglo one_hot
vars_w_categorized_month = categorizer.categorize_int(vars_ds, cat_col=MONTH_COL)
one_hot_month = categorizer.one_hot(vars_w_categorized_month, cat_col=MONTH_COL)

# convierto los dias previos no laborables en arreglos one_hot
vars_w_categorized_prevh = categorizer.categorize_int(vars_ds, cat_col=PREVH_COL)
one_hot_prevh = categorizer.one_hot(vars_w_categorized_prevh, cat_col=PREVH_COL)

# convierto los dias posteriores no laborables en arreglos one_hot
vars_w_categorized_posth = categorizer.categorize_int(vars_ds, cat_col=POSTH_COL)
one_hot_posth = categorizer.one_hot(vars_w_categorized_posth, cat_col=POSTH_COL)


final_vars = numpy.hstack((one_hot_dow, one_hot_dom, one_hot_month, one_hot_prevh, one_hot_posth))
PREVIOUS_ENTRIES_COUNT = 2
# defino el tensor de variables o entrada como la union de los tensores one_hot calculados antes.# armo el tensor final de entrada
# ademas le adjunto entradas de dias anteriores
__final_vars = final_vars
__final_demand = final_demand

for _ in range(PREVIOUS_ENTRIES_COUNT):
    __final_demand = __final_demand[:-1]
    __final_vars = numpy.hstack((__final_vars[1:], __final_demand))
# __final_demand = __final_demand[:-1]
# __final_vars = numpy.hstack((__final_vars[1:], __final_demand))

final_vars = __final_vars
# alineo la demanda con los valores de entrada
final_demand = final_demand[PREVIOUS_ENTRIES_COUNT:]

# defino el tensor de variables o entrada como la union de los tensores one_hot calculados antes.
# ademas le adjunto la demanda del dia anterior (por eso los one_hot comienzan en la fila 1 y el final_demand termina antes de la ultima fila)
# final_vars = numpy.hstack((one_hot_dow[1:], one_hot_dom[1:], one_hot_month[1:], one_hot_prevh[1:], one_hot_posth[1:], final_demand[:-1]))
# final_demand = final_demand[1:]  # alineo la demanda final con las variables de entrada

# CONSTRUCCION DE LA RED -----------------------------------------------------------------------------------------------------------

# Parameters
learning_rate = 0.001
training_epochs = 5000

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
# n_hidden_1 = int((train_in_ds.shape[1] + train_out_ds.shape[1]) / 2)
# n_hidden_2 = int((train_in_ds.shape[1] + train_out_ds.shape[1]) / 2)
n_hidden_1 = int(train_in_ds.shape[1] * 0.8)
n_hidden_2 = int(train_in_ds.shape[1] * 0.8)
n_hidden_3 = int(train_in_ds.shape[1] * 0.8)

WEIGHT_SEED = 11


def build_random_weight_tensor(in_size, out_size):
    """ Construye un tensor de in_size * out_size elementos con valores aleatorios siguiendo la distribucion normal.
    Cada valor representa el peso de la entrada 'i' hacia la neurona 'j' (siendo i una fila y j una columna) """
    return tf.random_normal(shape=[in_size, out_size], seed=WEIGHT_SEED)


BIAS_SEED = 7


def build_random_bias_tensor(out_size):
    """ Crea un tensor de umbrales de activacion para una capa de la red neuronal """
    return tf.random_normal(shape=[out_size], seed=BIAS_SEED)


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
    'hidden_3': tf.Variable(build_random_weight_tensor(n_hidden_2, n_hidden_3)),
    'out': tf.Variable(build_random_weight_tensor(n_hidden_3, n_classes))
}

# defino los umbrales de activacion
biases = {
    'bias_1': tf.Variable(build_random_bias_tensor(n_hidden_1)),
    'bias_2': tf.Variable(build_random_bias_tensor(n_hidden_2)),
    'bias_3': tf.Variable(build_random_bias_tensor(n_hidden_3)),
    'out': tf.Variable(build_random_bias_tensor(n_classes))
}


# Create model
def multilayer_perceptron(x):
    # Cada capa resuelve: x * w + b
    # Conecto capa de entrada con capa oculta 1
    layer_1 = tf.matmul(x, weights['hidden_1']) + biases['bias_1']
    layer_1 = tf.nn.sigmoid(layer_1)
    # Conecto capa oculta 1 con capa oculta 2
    layer_2 = tf.matmul(layer_1, weights['hidden_2']) + biases['bias_2']
    layer_2 = tf.nn.sigmoid(layer_2)
    # Conecto capa oculta 2 con capa oculta 3
    layer_3 = tf.matmul(layer_2, weights['hidden_3']) + biases['bias_3']
    layer_3 = tf.nn.sigmoid(layer_3)
    # Conecto capa oculta 3 con capa de salida
    out_layer = tf.matmul(layer_3, weights['out']) + biases['out']
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
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))  # determino punto a punto si la prediccion es correcta

accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print("Accuracy:", accuracy.eval({X: test_in_ds, Y: test_out_ds}, session=sess))


def translate_prediction(pred_value, cat_count=CAT_COUNT):
    a = numpy.zeros(CAT_COUNT)
    a[pred_value] = 1
    return a


test_in_0 = test_in_ds[0]
prediction_0 = sess.run(tf.argmax(pred, 1), feed_dict={X: [test_in_0]})
one_hot_prediction_0 = translate_prediction(prediction_0)

test_in_1 = test_in_ds[1]
test_in_1[RAW_IN_COL_COUNT:RAW_IN_COL_COUNT + CAT_COUNT] = one_hot_prediction_0

prediction = sess.run(tf.argmax(pred, 1), feed_dict={X: test_in_ds})
true_values = sess.run(tf.argmax(Y, 1), feed_dict={Y: test_out_ds})

plt.plot(prediction)
plt.plot(true_values)
axes = plt.gca()
axes.set_ylim([0, CAT_COUNT])  # seteo limite en el eje y entre 0 y 1
plt.show()

sess.close()
