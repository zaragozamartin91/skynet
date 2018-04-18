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

def normalize_dataset(dataset):
    """ Normaliza un dataset """
    col_count = dataset.shape[1]
    for col in range(col_count):
        sub_ds = dataset[:, col]
        n_ds = sub_ds / float(max(sub_ds))
        dataset[:, col] = n_ds
    return dataset


def de_normalize_dataset(normalized_ds, original_ds):
    """ 
    Des-normaliza un dataset a partir del dataset original (completo) 
    :param normalized_ds : Dataset normalizado a des-normalizar
    :param original_ds : Dataset ORIGINAL (no particionado) del cual obtener los valores originales
    :return : Dataset des-normalizado
    """
    col_count = original_ds.shape[1]
    for col in range(col_count):
        original_sub_ds = original_ds[:, col]
        min_value = min(original_sub_ds)
        max_value = max(original_sub_ds)
        a = normalized_ds[:, col] * max_value
        b = a + min_value
        normalized_ds[:, col] = b
    return normalized_ds


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


def append_prev_demand(vars_ds, demand_ds):
    """ Crea un nuevo dataset que contenga los valores de las variables de entrada y dos columnas extra que representan las demandas del dia anterior
    de entrada y salida respectivamente """
    in_demands = [(0, )]
    out_demands = [(0, )]
    row_count = len(vars_ds)
    for i in range(row_count):
        if (i == 0): continue
        prev_entry = demand_ds[i - 1]
        in_demands.append((prev_entry[0], ))
        out_demands.append((prev_entry[1], ))
    return numpy.hstack((vars_ds, numpy.array(in_demands), numpy.array(out_demands)))


def build_win_matrix(dataset, win_size=1):
    """ 
    Crea una matriz tridimensional de dimension (x,y,z)
    x representa las entradas
    y representa los timesteps (o memoria de la matriz y red)
    z representa las columnas o categorias de datos
    """
    win_matrix = []
    row_count = len(dataset)
    for idx in range(win_size , row_count):
        lower_limit = idx - win_size
        entries = dataset[lower_limit:idx]
        row = []
        for entry in entries:
            row.append(entry)
        win_matrix.append(row)
    # win_matrix.reverse()
    return numpy.array(win_matrix)

numpy.random.seed(7)

# COLUMNAS:
#  0     1   2    3     4         5      6         7          8      9
# index,dow,dom,month,year,in_demand,out_demand,prev_holy,pos_holy,minfl
vars_df = pandas.read_csv('full_data.csv', usecols=[1, 2, 3, 7, 8])
demand_df = pandas.read_csv('full_data.csv', usecols=[5, 6])

#  0     1   2    3     4         5      6         7          8      9
# index,dow,dom,month,year,in_demand,out_demand,prev_holy,pos_holy,minfl
dates_ds = pandas.read_csv('full_data.csv', usecols=[2, 3, 4]).values

vars_ds = vars_df.values.astype('float64')
demand_ds = demand_df.values.astype('float64')

# Modifico el dataset de variables de entrada para incorporar las columnas de la demanda de entrada y salida del dia anterior
vars_ds = append_prev_demand(vars_ds, demand_ds)

# normalizo los datasets de variables de entrada y las demandas en sus respectivos dominios
normalize_dataset(vars_ds)
normalize_dataset(demand_ds)

column_count = vars_ds.shape[1] # Cantidad de columnas del dataset de entrada
timesteps = 15 # cantidad de pasos memoria
vars_ds = build_win_matrix(vars_ds , timesteps)
# hago coincidir los dias y las demandas con la nueva matriz de ventana
dates_ds = dates_ds[timesteps:]
demand_ds = demand_ds[timesteps:]

# reshape hacia (CANT_FILAS, CANT_ESTADOS = 1 , CANT_COLUMNAS)
# este reshape se hace para trabajar con LSTM con timesteps == 1
# vars_ds = vars_ds.reshape((len(vars_ds), 1, column_count))


# train_size = int(len(vars_ds) * 0.67)
# test_size = len(vars_ds) - train_size

train_size = 675
# test_size = 365
test_size = 30


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

# input_dim = train_vars.shape[1] # la cantidad de neuronas de input es igual a la cantidad de columnas del dataset de entrada
model = Sequential()

# keras.layers.LSTM(units, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, implementation=1, return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False)

# batch_size = 5
batch_size = gcd(train_size , test_size)

model.add(LSTM(4, stateful=True, batch_input_shape=(batch_size, timesteps, column_count)))
# model.add(LSTM(50, input_shape=(1,vars_ds.shape[2]) , stateful=True, batch_input_shape=(batch_size,1,vars_ds.shape[2])) )
model.add(Dense(30, activation='relu'))
model.add(Dense(2, activation='relu'))
opt = optimizers.adam(lr=0.001)
# model.compile(loss='binary_crossentropy', optimizer=opt)
model.compile(loss='mean_squared_error', optimizer=opt)
model.fit(train_vars, train_demand, epochs=200, batch_size=batch_size, shuffle=False, verbose=2)


# trainPredict = model.predict(train_vars)
predicted = model.predict(test_vars, batch_size=batch_size)
denormalized_predicted = de_normalize_dataset(predicted.copy(), demand_df.values)



true_dates = dates_ds[test_lower_limit:test_upper_limit]  # obtengo las fechas a graficar
last_date_idx = test_size - 1
start_date = date.today().replace(true_dates[0, 2], true_dates[0, 1], true_dates[0, 0])  # obtengo la fecha de inicio
end_date = date.today().replace(true_dates[last_date_idx, 2], true_dates[last_date_idx, 1], true_dates[last_date_idx, 0])  # obtengo la fecha de fin
all_ticks = numpy.linspace(date2num(start_date), date2num(end_date), test_size)  # obtengo un arreglo con todos los valores numericos de fechas
tick_spacing = 30
# major_ticks = numpy.arange(date2num(start_date), date2num(end_date), tick_spacing)  # obtengo un arreglo con los valores de fecha que quiero mostrar
major_ticks = numpy.linspace(date2num(start_date), date2num(end_date), tick_spacing)  # obtengo un arreglo con los valores de fecha que quiero mostrar
# major_tick_labels = [date.strftime("%Y-%m") for date in num2date(major_ticks)]
major_tick_labels = [date.strftime("%m-%d") for date in num2date(major_ticks)]

# PLOTEO DE LA DEMANDA DE SALIDA JUNTO CON LA PORCION DE INCUMBENCIA DE LOS DATOS ORIGINALES -----------------------------------------
true_out_demand = demand_ds[test_lower_limit:test_upper_limit, 1]
predicted_out_demand = predicted[:, 1]
plot_w_xticks(all_ticks, major_ticks, major_tick_labels, [(true_out_demand, 'b'), (predicted_out_demand, 'r')])
axes = plt.gca()
axes.set_ylim([0, 1])  # seteo limite en el eje y entre 0 y 1
plt.show()

# PLOTEO LA DIFERENCIA ABSOLUTA ENTRE VALORES REALES Y LOS VALORES PREDECIDOS NORMALIZADOS -----------------------------------------
# error_ds = true_out_demand - predicted_out_demand
# error_ds = abs(error_ds)
# plot_w_xticks(all_ticks, major_ticks, major_tick_labels, [(error_ds, 'b')])
# axes = plt.gca()
# axes.set_ylim([0, 1])  # seteo limite en el eje y entre 0 y 1
# plt.show()




denormalized_true_out_demand = de_normalize_dataset(test_demand.copy() , demand_df.values)[:,1]
denormalized_predicted_out_demand = denormalized_predicted[:, 1] 

# PLOTEO EN MILES DE PESOS (DES-NORMALIZADO)------------------------------------------------------------------------------------------

plot_w_xticks(all_ticks, major_ticks, major_tick_labels, [(denormalized_true_out_demand/1000, 'b'), (denormalized_predicted_out_demand/1000, 'r')])
plt.show()


diff = true_out_demand - predicted_out_demand
diff = abs(diff)
error_ds = diff / true_out_demand
plot_w_xticks(all_ticks, major_ticks, major_tick_labels, [(error_ds, 'b')])
axes = plt.gca()
axes.set_ylim([0, 1])  # seteo limite en el eje y entre 0 y 1
plt.show()



# PLOTEO EL ERROR COMETIDO ----------------------------------------------------------------------------------------------------------
# ERROR USANDO LOS VALORES NORMALIZADOS
# diff = true_out_demand - predicted_out_demand
# diff = abs(diff)
# plus_one = true_out_demand + 1
# error_ds = diff / plus_one
# plot_w_xticks(all_ticks, major_ticks, major_tick_labels, [(error_ds, 'b')])
# axes = plt.gca()
# axes.set_ylim([0, 1])  # seteo limite en el eje y entre 0 y 1
# plt.show()

# ERROR USANDO LOS VALORES DES-NORMALIZADOS
diff = denormalized_true_out_demand - denormalized_predicted_out_demand
diff = abs(diff)
plus_one = denormalized_true_out_demand + 1
error_ds = diff / plus_one
# error_ds = diff / denormalized_true_out_demand
plot_w_xticks(all_ticks, major_ticks, major_tick_labels, [(error_ds, 'b')])
axes = plt.gca()
axes.set_ylim([0, 1])  # seteo limite en el eje y entre 0 y 1
plt.show()

# PLOTEO LOS VALORES PREDECIDOS Y LOS ORIGINALES DES-NORMALIZADOS -----------------------------------------------------------------------------
graph = plot_w_xticks(all_ticks, major_ticks, major_tick_labels, [(denormalized_true_out_demand / 1000, 'b'), (denormalized_predicted_out_demand / 1000, 'r')])
graph.set_ylabel('Dinero en MILES')
plt.show()