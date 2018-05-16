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

from skmatrix import categorizer


def normalize_dataset(dataset):
    """ Normaliza un dataset """
    col_count = dataset.shape[1]
    for col in range(col_count):
        sub_ds = dataset[:, col]
        max_value = float(max(sub_ds))
        min_value = float(min(sub_ds))
        diff = max_value - min_value
        n_ds = sub_ds - min_value
        n_ds = n_ds / diff
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
        diff = max_value - min_value
        a = normalized_ds[:, col] * diff
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
    """ Crea un nuevo dataset que contenga los valores de las variables de entrada y una columna extra que representa la demanda neta del dia anterior """
    in_demands = [(0, )]
    row_count = len(vars_ds)
    for i in range(row_count):
        if (i == 0): continue
        prev_entry = demand_ds[i - 1]
        in_demands.append((prev_entry[0], ))
    return numpy.hstack((vars_ds, numpy.array(in_demands)))


def build_win_matrix(dataset, win_size):
    """ 
    Crea una matriz tridimensional en la que cada entrada contiene los datos de ese dia y los datos de los win_size dias anteriores  
    """
    win_matrix = []
    row_count = len(dataset)
    for idx in range(win_size, row_count):
        lower_limit = idx - win_size
        entries = dataset[lower_limit:idx + 1]
        row = []
        for entry in entries:
            row.append(entry)
        win_matrix.append(row)
    return numpy.array(win_matrix)


numpy.random.seed(7)

input_file = 'full_entrada_salida_pesos_506.csv'

# COLUMNAS:
#  0     1   2    3     4         5      6         7          8      9
# index,dow,dom,month,year,in_demand,out_demand,prev_holi,pos_holi,minfl
vars_df = pandas.read_csv(input_file, usecols=[1, 2, 3, 7, 8])
demand_df = pandas.read_csv(input_file, usecols=[5, 6])

DIFF_DEMAND = demand_df.values[:,0] - demand_df.values[:,1]
DIFF_DEMAND.resize( (len(DIFF_DEMAND) , 1) )

#  0     1   2    3     4         5      6         7          8      9
# index,dow,dom,month,year,in_demand,out_demand,prev_holi,pos_holi,minfl
dates_ds = pandas.read_csv(input_file, usecols=[2, 3, 4]).values

vars_ds = vars_df.values.astype('float64')
demand_ds = demand_df.values.astype('float64')

in_demand = demand_ds[:, 0]
out_demand = demand_ds[:, 1]
demand_ds = in_demand - out_demand
demand_ds.resize((len(demand_ds), 1))

# obtengo los valores de demanda categorizados
CAT_COUNT = 30
categorized_demand, demand_bags = categorizer.categorize_real_w_equal_frames(demand_ds, cat_count=CAT_COUNT)
demand_ds = categorized_demand.copy()

# Agrego los valores de la demanda de entrada y salida previas
vars_ds = append_prev_demand(vars_ds, demand_ds)
# Descarto el primer dia dado que no cuenta con demanda previa o su demanda previa es un falso 0.0
vars_ds = vars_ds[1:]
demand_ds = demand_ds[1:]
dates_ds = dates_ds[1:]

# Asigno valores de 0 a 1 a todas las entradas
normalize_dataset(vars_ds)
normalize_dataset(demand_ds)

column_count = vars_ds.shape[1]  # Cantidad de columnas del dataset de entrada
timesteps = 10  # cantidad de pasos memoria
vars_ds = build_win_matrix(vars_ds, timesteps)
# hago coincidir los dias y las demandas con la nueva matriz de ventana
demand_ds = demand_ds[timesteps:]
dates_ds = dates_ds[timesteps:]

# reshape hacia (CANT_FILAS, CANT_ESTADOS = 1 , CANT_COLUMNAS)
# este reshape se hace para trabajar con LSTM con timesteps == 1
# vars_ds = vars_ds.reshape((len(vars_ds), 1, column_count))

# train_size = int(len(vars_ds) * 0.67)
# test_size = len(vars_ds) - train_size

test_size = 30
train_size = len(vars_ds) - test_size
# train_size = 675

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
true_dates = dates_ds[test_lower_limit:test_upper_limit]  # obtengo las fechas a graficar

# Keras de manera implicita trabaja siempre con una capa de entrada
# input_dim determina la cantidad de neuronas de la capa de entrada
# el primer parametro de 'Dense' es la cantidad de neuronas de la capa oculta
# una red densa o Dense es aquella en la que todas las neuronas de una capa N estan conectadas con todas las de la capa N+1
# el modelo a crear es Multilayer Perceptron
# la cantidad de neuronas de la capa de salida debe coincidir con la cantidad de valores a predecir
# Si la ultima capa tiene una funcion de activacion, entonces estamos modelando un problema de CLASIFICACION / CLUSTERIZACION en vez de uno de PREDICCION
# epochs es el número de pasadas por todo el conjunto de datos de entrenamiento
# batch_size es el número de muestras que se usan para calcular una actualización de los pesos
# LA CAPA N+1 NO DEBERIA SER MAS GRANDE QUE LA CAPA N

# input_dim = train_vars.shape[1] # la cantidad de neuronas de input es igual a la cantidad de columnas del dataset de entrada
model = Sequential()

# keras.layers.LSTM(units, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, implementation=1, return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False)

# batch_size = 5
# batch_size = gcd(train_size , test_size)
batch_size = 1
epochs = 200
model.add(LSTM(30, stateful=True, batch_input_shape=(batch_size, timesteps + 1, column_count)))
# model.add(LSTM(50, input_shape=(1,vars_ds.shape[2]) , stateful=True, batch_input_shape=(batch_size,1,vars_ds.shape[2])) )
model.add(Dense(30, activation='relu'))
model.add(Dense(1, activation='relu'))
opt = optimizers.adam(lr=0.001)
model.compile(loss='binary_crossentropy', optimizer=opt)
# model.compile(loss='mean_squared_error', optimizer=opt)
model.fit(train_vars, train_demand, epochs=epochs, batch_size=batch_size, shuffle=False, verbose=2)

idx = 0
predicted = []

# idx = 0
test_var = test_vars[idx:idx + 1]
predicted_entry = model.predict(test_var, batch_size=batch_size).reshape((1, ))
predicted.append(predicted_entry)
idx += 1

for idx in range(1, test_size):
    test_var = test_vars[idx:idx + 1]  # obtengo la entrada sobre la cual voy a hacer la prediccion
    start_ts = 0 if (timesteps - idx + 1 <= 0) else (timesteps - idx + 1)
    end_ts = timesteps + 1 + 1  # si recuerdo 10 dias , entonces tengo 11 elementos (10 pasados + 1 actual)
    start_pred = 0 if (idx - timesteps - 1 <= 0) else idx - timesteps - 1
    end_pred = idx
    test_var[0, start_ts:end_ts, 5:] = predicted[start_pred:end_pred]
    predicted_entry = model.predict(test_var, batch_size=batch_size).reshape((1, ))
    predicted.append(predicted_entry)
    print('start_ts: %d , end_ts: %d , start_pred: %d , end_pred: %d' % (start_ts, end_ts, start_pred, end_pred))

predicted = numpy.array(predicted)


last_date_idx = test_size - 1
start_date = date.today().replace(true_dates[0, 2], true_dates[0, 1], true_dates[0, 0])  # obtengo la fecha de inicio
end_date = date.today().replace(true_dates[last_date_idx, 2], true_dates[last_date_idx, 1], true_dates[last_date_idx, 0])  # obtengo la fecha de fin
all_ticks = numpy.linspace(date2num(start_date), date2num(end_date), test_size)  # obtengo un arreglo con todos los valores numericos de fechas

tick_spacing = test_size if test_size <= 60 else 12
date_format = "%m/%d" if test_size <= 60 else "%y/%m"

# major_ticks = numpy.arange(date2num(start_date), date2num(end_date), tick_spacing)  # obtengo un arreglo con los valores de fecha que quiero mostrar
major_ticks = numpy.linspace(date2num(start_date), date2num(end_date), tick_spacing)  # obtengo un arreglo con los valores de fecha que quiero mostrar
major_tick_labels = [date.strftime(date_format) for date in num2date(major_ticks)]

# PLOTEO DE LA DEMANDA DE SALIDA NORMALIZADA JUNTO CON LA PORCION DE INCUMBENCIA DE LOS DATOS ORIGINALES -----------------------------------------
true_net_demand = test_demand
predicted_net_demand = predicted
plot_w_xticks(all_ticks, major_ticks, major_tick_labels, [(true_net_demand, 'b-o'), (predicted_net_demand, 'r-o')])
axes = plt.gca()
# axes.set_ylim([0, 1])  # seteo limite en el eje y entre 0 y 1
plt.show()

