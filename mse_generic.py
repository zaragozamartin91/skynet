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
        max_value = float(max(sub_ds))
        min_value = float(min(sub_ds))
        diff = max_value -  min_value
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


numpy.random.seed(7)

input_file = 'full_entrada_salida_pesos_151.csv'

# COLUMNAS:
#  0     1   2    3     4         5      6         7          8      9
# index,dow,dom,month,year,in_demand,out_demand,prev_holi,pos_holi,minfl
vars_df = pandas.read_csv(input_file, usecols=[1, 2, 3, 7, 8])
demand_df = pandas.read_csv(input_file, usecols=[5, 6])

DIFF_DEMAND = demand_df.values[:, 0] - demand_df.values[:, 1]
DIFF_DEMAND.resize((len(DIFF_DEMAND), 1))

#  0     1   2    3     4         5      6         7          8      9
# index,dow,dom,month,year,in_demand,out_demand,prev_holi,pos_holi,minfl
dates_ds = pandas.read_csv(input_file, usecols=[2, 3, 4]).values

vars_ds = vars_df.values.astype('float64')
demand_ds = demand_df.values.astype('float64')

# reasigno a demand_ds como la demanda neta (entrada - salida)
in_demand = demand_ds[:, 0]
out_demand = demand_ds[:, 1]
demand_ds = in_demand - out_demand
demand_ds.resize((len(demand_ds), 1))

# Asigno valores de 0 a 1 a todas las entradas
normalize_dataset(vars_ds)
normalize_dataset(demand_ds)

column_count = vars_ds.shape[1]  # Cantidad de columnas del dataset de entrada

# defino la cantidad de registros de entrenamiento y de prueba a partir del batch_size
batch_size = 5
test_size = 30
train_size = batch_size * int((len(vars_ds) - test_size) / batch_size)

train_lower_limit = 0
train_upper_limit = train_size
test_lower_limit = train_upper_limit
test_upper_limit = train_size + test_size

# defino las variables y valores de demanda de entrenamiento y de prueba
train_vars = vars_ds[train_lower_limit:train_upper_limit]
train_demand = demand_ds[train_lower_limit:train_upper_limit]
train_dates = dates_ds[train_lower_limit:train_upper_limit]

test_vars = vars_ds[test_lower_limit:test_upper_limit]
test_demand = demand_ds[test_lower_limit:test_upper_limit]
test_dates = dates_ds[test_lower_limit:test_upper_limit]


epochs = 200

input_dim = column_count  # la cantidad de neuronas de input es igual a la cantidad de columnas del dataset de entrada
model = Sequential()
model.add(Dense(50, input_dim=input_dim, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(1, activation='relu'))
opt = optimizers.adam(lr=0.001)
# model.compile(loss='binary_crossentropy', optimizer=opt)
model.compile(loss='mean_squared_error', optimizer=opt)
model.fit(train_vars, train_demand, epochs=epochs, batch_size=batch_size, verbose=2)


predicted = model.predict(test_vars)


denormalized_predicted = de_normalize_dataset(predicted.copy(), DIFF_DEMAND)


true_dates = test_dates
last_date_idx = test_size - 1
start_date = date.today().replace(true_dates[0, 2], true_dates[0, 1], true_dates[0, 0])  # obtengo la fecha de inicio
end_date = date.today().replace(true_dates[last_date_idx, 2], true_dates[last_date_idx, 1], true_dates[last_date_idx, 0])  # obtengo la fecha de fin
all_ticks = numpy.linspace(date2num(start_date), date2num(end_date), test_size)  # obtengo un arreglo con todos los valores numericos de fechas

tick_spacing = test_size if test_size <= 60 else 30
date_format = "%m/%d" if test_size <= 60 else "%y/%m"

# major_ticks = numpy.arange(date2num(start_date), date2num(end_date), tick_spacing)  # obtengo un arreglo con los valores de fecha que quiero mostrar
major_ticks = numpy.linspace(date2num(start_date), date2num(end_date), tick_spacing)  # obtengo un arreglo con los valores de fecha que quiero mostrar
major_tick_labels = [date.strftime(date_format) for date in num2date(major_ticks)]

# PLOTEO DE LA DEMANDA DE SALIDA NORMALIZADA JUNTO CON LA PORCION DE INCUMBENCIA DE LOS DATOS ORIGINALES -----------------------------------------
true_net_demand = test_demand
predicted_net_demand = predicted
plot_w_xticks(all_ticks, major_ticks, major_tick_labels, [(true_net_demand, 'b-o'), (predicted_net_demand, 'r-o')])
axes = plt.gca()
#axes.set_ylim([0, 1])  # seteo limite en el eje y entre 0 y 1
plt.grid()
plt.show()

# PLOTEO EL ERROR COMETIDO ----------------------------------------------------------------------------------------------------------
denormalized_true_net_demand = de_normalize_dataset(test_demand.copy(), DIFF_DEMAND)
denormalized_predicted_net_demand = denormalized_predicted[:test_size]

# ERROR USANDO LOS VALORES NORMALIZADOS
diff = true_net_demand - predicted_net_demand
diff = abs(diff)
plus_one = true_net_demand
error_ds = diff / plus_one
graph = plot_w_xticks(all_ticks, major_ticks, major_tick_labels, [(error_ds, 'b-o')])
graph.set_title('Error con valores normalizados')
axes = plt.gca()
axes.set_ylim([0, 1])  # seteo limite en el eje y entre 0 y 1
plt.show()


# ERROR USANDO LOS VALORES DES-NORMALIZADOS
diff = denormalized_true_net_demand - denormalized_predicted_net_demand
diff = abs(diff)
plus_one = abs(denormalized_true_net_demand) + 1
error_ds = diff / plus_one
# # error_ds = diff / denormalized_true_net_demand
graph = plot_w_xticks(all_ticks, major_ticks, major_tick_labels, [(error_ds, 'b-o')])
graph.set_title('Error con valores DES-normalizados')
axes = plt.gca()
axes.set_ylim([0, 10])  # seteo limite en el eje y entre 0 y 1
plt.show()

# PLOTEO LOS VALORES PREDECIDOS Y LOS ORIGINALES DES-NORMALIZADOS (en miles de pesos) -----------------------------------------------------------------------------
graph = plot_w_xticks(all_ticks, major_ticks, major_tick_labels, [(denormalized_true_net_demand / 1000, 'b-o'),
                                                                  (denormalized_predicted_net_demand / 1000, 'r-o')])
graph.set_ylabel('Dinero en MILES')
plt.grid()
plt.show()
