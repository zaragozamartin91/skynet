import numpy
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras import optimizers

from datetime import date
from datetime import datetime

from matplotlib.dates import date2num
from matplotlib.dates import num2date

from fractions import gcd

from skmatrix import normalizer


def normalize_dataset(dataset):
    return normalizer.normalize_dataset(dataset)


def de_normalize_dataset(normalized_ds, original_ds):
    return normalizer.de_normalize_dataset(normalized_ds, original_ds)


def plot_w_xticks(all_xticks, major_xticks, major_xticks_labels, yplots):
    """ 
    Plotea sets de datos
    :param all_xticks : todos los valores del eje x
    :param major_xticks : los valores principales del eje x (valores a mostrar)
    :param major_xticks_labels : labels de los principales valores del eje x (arreglo de strings)
    :param yplots : arreglo de tuplas '(datos_eje_y , color_ploteo)'
    """
    fig = plt.figure()
    graph = fig.add_subplot(111)
    x = all_xticks
    for yplot in yplots:
        graph.plot(x, yplot[0], yplot[1])
    graph.set_xticks(major_xticks)
    graph.set_xticklabels(major_xticks_labels)
    return graph


def get_timestep_demands(vars_ds, demand_ds, timesteps):
    """ Crea un nuevo dataset que contenga los valores de las variables de entrada y dos columnas extra que representan las demandas del dia anterior
    de entrada y salida respectivamente """
    in_demands = []
    out_demands = []
    row_count = len(vars_ds)
    for i in range(timesteps, row_count):
        previous_demands = demand_ds[i - timesteps:i]
        in_demands.append(previous_demands[:, 0])
        out_demands.append(previous_demands[:, 1])
    return (numpy.array(in_demands) , numpy.array(out_demands))

numpy.random.seed(7)

input_file = 'full_entrada_salida_pesos_100.csv'

# COLUMNAS:
#  0     1   2    3     4         5      6         7          8      9
# index,dow,dom,month,year,in_demand,out_demand,prev_holy,pos_holy,minfl
vars_df = pandas.read_csv(input_file, usecols=[1, 2, 3, 7, 8])
demand_df = pandas.read_csv(input_file, usecols=[5, 6])

dates_ds = pandas.read_csv(input_file, usecols=[2, 3, 4]).values

vars_ds = vars_df.values.astype('float64')
demand_ds = demand_df.values.astype('float64')


timesteps = 5
in_demands , out_demands = get_timestep_demands(vars_ds, demand_ds, timesteps)
vars_ds = numpy.hstack( ( vars_ds[timesteps:] , in_demands , out_demands ) )
# pongo en fase los valores de las demandas y de las fechas
demand_ds = demand_ds[timesteps:]
dates_ds = dates_ds[timesteps:]

# Asigno valores de 0 a 1 a todas las entradas
normalize_dataset(vars_ds)
normalize_dataset(demand_ds)

column_count = vars_ds.shape[1]  # Cantidad de columnas del dataset de entrada


test_size = 60
train_size = 600

train_lower_limit = 0
train_upper_limit = train_size
test_lower_limit = train_size
test_upper_limit = train_size + test_size

# Creo el dataset de entrenamiento. EL mismo toma las filas
# del 0 a train_size (no inclusive) y todas las columnas (rango ':')
train_vars = vars_ds[train_lower_limit:train_upper_limit]
train_demand = demand_ds[train_lower_limit:train_upper_limit]

test_vars = vars_ds[test_lower_limit:test_upper_limit]
test_demand = demand_ds[test_lower_limit:test_upper_limit]

# Keras de manera implicita trabaja siempre con una capa de entrada
# input_dim determina la cantidad de neuronas de la capa de entrada
# el primer parametro de 'Dense' es la cantidad de neuronas de la capa oculta
# una red densa o Dense es aquella en la que todas las neuronas de una capa N estan conectadas con todas las de la capa N+1
# el modelo a crear es Multilayer Perceptron
# la cantidad de neuronas de la capa de salida debe coincidir con la cantidad de valores a predecir
# Si la ultima capa tiene una funcion de activacion, entonces estamos modelando un problema de CLASIFICACION / CLUSTERIZACION en vez de uno de PREDICCION
# epochs es el número de pasadas por todo el conjunto de datos de entrenamiento
# batch_size es el número de muestras que se usan para calcular una actualización de los pesos

epochs = 300

input_dim = column_count  # la cantidad de neuronas de input es igual a la cantidad de columnas del dataset de entrada
model = Sequential()
model.add(Dense(80, input_dim=input_dim, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(2, activation='relu'))
opt = optimizers.adam(lr=0.001)
# model.compile(loss='binary_crossentropy', optimizer=opt)
model.compile(loss='mean_squared_error', optimizer=opt)
model.fit(train_vars, train_demand, epochs=epochs, batch_size=5, verbose=2)

# trainPredict = model.predict(train_vars)
predicted = model.predict(test_vars)

true_dates = dates_ds[test_lower_limit:test_upper_limit]  # obtengo las fechas a graficar
last_date_idx = test_size - 1
start_date = date.today().replace(true_dates[0, 2], true_dates[0, 1], true_dates[0, 0])  # obtengo la fecha de inicio
end_date = date.today().replace(true_dates[last_date_idx, 2], true_dates[last_date_idx, 1], true_dates[last_date_idx, 0])  # obtengo la fecha de fin
all_ticks = numpy.linspace(date2num(start_date), date2num(end_date), test_size)  # obtengo un arreglo con todos los valores numericos de fechas

tick_spacing = test_size if test_size <= 60 else 12
date_format = "%m-%d" if test_size <= 60 else "%y-%m"

# major_ticks = numpy.arange(date2num(start_date), date2num(end_date), tick_spacing)  # obtengo un arreglo con los valores de fecha que quiero mostrar
major_ticks = numpy.linspace(date2num(start_date), date2num(end_date), tick_spacing)  # obtengo un arreglo con los valores de fecha que quiero mostrar
major_tick_labels = [date.strftime(date_format) for date in num2date(major_ticks)]

# PLOTEO DE LA DEMANDA DE SALIDA JUNTO CON LA PORCION DE INCUMBENCIA DE LOS DATOS ORIGINALES -----------------------------------------
true_out_demand = demand_ds[test_lower_limit:test_upper_limit, 1]
predicted_out_demand = predicted[:, 1]
plot_w_xticks(all_ticks, major_ticks, major_tick_labels, [(true_out_demand, 'b-o'), (predicted_out_demand, 'r-o')])
axes = plt.gca()
axes.set_ylim([0, 1])  # seteo limite en el eje y entre 0 y 1
plt.show()

# PLOTEO DEL ERROR ---------------------------------------------------------------------------------------------------
diff = true_out_demand - predicted_out_demand
diff = abs(diff)
error_ds = diff / (predicted_out_demand + 0.0001)
plt.plot(error_ds, 'r-o')
axes = plt.gca()
axes.set_ylim([0, 1]) # seteo limite en el eje y entre 0 y 1
plt.show()
