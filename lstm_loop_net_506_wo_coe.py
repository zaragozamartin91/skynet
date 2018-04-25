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

# QUITO LAS ENTRADAS QUE GENERAN MUCHO RUIDO
del_count = 0
DIFF_DEMAND = numpy.delete(DIFF_DEMAND, 175 - del_count, axis=0)
demand_ds = numpy.delete(demand_ds, 175 - del_count, axis=0)
vars_ds = numpy.delete(vars_ds, 175 - del_count, axis=0)
dates_ds = numpy.delete(dates_ds, 175 - del_count, axis=0)
del_count += 1
DIFF_DEMAND = numpy.delete(DIFF_DEMAND, 176 - del_count, axis=0)
demand_ds = numpy.delete(demand_ds, 176 - del_count, axis=0)
vars_ds = numpy.delete(vars_ds, 176 - del_count, axis=0)
dates_ds = numpy.delete(dates_ds, 176 - del_count, axis=0)
del_count += 1
DIFF_DEMAND = numpy.delete(DIFF_DEMAND, 181 - del_count, axis=0)
demand_ds = numpy.delete(demand_ds, 181 - del_count, axis=0)
vars_ds = numpy.delete(vars_ds, 181 - del_count, axis=0)
dates_ds = numpy.delete(dates_ds, 181 - del_count, axis=0)
del_count += 1
DIFF_DEMAND = numpy.delete(DIFF_DEMAND, 198 - del_count, axis=0)
demand_ds = numpy.delete(demand_ds, 198 - del_count, axis=0)
vars_ds = numpy.delete(vars_ds, 198 - del_count, axis=0)
dates_ds = numpy.delete(dates_ds, 198 - del_count, axis=0)
del_count += 1
DIFF_DEMAND = numpy.delete(DIFF_DEMAND, 199 - del_count, axis=0)
demand_ds = numpy.delete(demand_ds, 199 - del_count, axis=0)
vars_ds = numpy.delete(vars_ds, 199 - del_count, axis=0)
dates_ds = numpy.delete(dates_ds, 199 - del_count, axis=0)

# Agrego los valores de la demanda de entrada y salida previas
vars_ds = append_prev_demand(vars_ds, demand_ds)
# Limite de datos donde no hay demasiado ruido
DATASET_UPPER_LIMIT = 800 - del_count
# Descarto el primer dia dado que no cuenta con demanda previa o su demanda previa es un falso 0.0
# tambien descarto los registros posteriores al 800 dado que introducen demasiado ruido
vars_ds = vars_ds[1:DATASET_UPPER_LIMIT]
demand_ds = demand_ds[1:DATASET_UPPER_LIMIT]
dates_ds = dates_ds[1:DATASET_UPPER_LIMIT]
DIFF_DEMAND = DIFF_DEMAND[1:DATASET_UPPER_LIMIT]


# Asigno valores de 0 a 1 a todas las entradas
normalize_dataset(vars_ds)
normalize_dataset(demand_ds)

column_count = vars_ds.shape[1]  # Cantidad de columnas del dataset de entrada
timesteps = 10  # cantidad de pasos memoria
# construyo la matriz con memoria y hago coincidir los dias y las demandas con la nueva matriz de ventana
vars_ds = build_win_matrix(vars_ds, timesteps)
demand_ds = demand_ds[timesteps:]
dates_ds = dates_ds[timesteps:]


test_size = 22
# El registro donde las predicciones de coe comienzan es el 481 (numero original)
# - 1 (descartamos el primer registro dado que desestimamos la demanda del primer dia)
# - timesteps - 1 (desestimamos los primeros 'timesteps' registros)
# - cantidad de registros ruidosos eliminados
COE_START_IDX = 482 - 1 - timesteps - 1 - del_count
# El registro inmediato despues de las predicciones de COE esta a 22 regs de distancia que aquel del comienzo
COE_END_IDX = COE_START_IDX + test_size


# separo los sets de datos en ANTES y DESPUES del registro inicial de COE
vars_ds_before_coe = vars_ds[:COE_START_IDX]
demand_ds_before_coe = demand_ds[:COE_START_IDX]
dates_ds_before_coe = dates_ds[:COE_START_IDX]

vars_ds_after_coe = vars_ds[COE_END_IDX + timesteps:]
demand_ds_after_coe = demand_ds[COE_END_IDX + timesteps:]
dates_ds_after_coe = dates_ds[COE_END_IDX + timesteps:]

train_vars = numpy.vstack( (vars_ds_before_coe , vars_ds_after_coe) )
train_demand = numpy.vstack( (demand_ds_before_coe , demand_ds_after_coe) )

# tomo como valores de testeo a aquellos que se encuentran en el rango de prediccion de COE
test_vars = vars_ds[COE_START_IDX:COE_END_IDX]
test_demand = demand_ds[COE_START_IDX:COE_END_IDX]
test_dates = dates_ds[COE_START_IDX:COE_END_IDX]



model = Sequential()

batch_size = 1
epochs = 5
model.add(LSTM(300, stateful=True, batch_input_shape=(batch_size, timesteps + 1, column_count)))
# model.add(LSTM(50, input_shape=(1,vars_ds.shape[2]) , stateful=True, batch_input_shape=(batch_size,1,vars_ds.shape[2])) )
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='relu'))
opt = optimizers.adam(lr=0.0001)
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

# trainPredict = model.predict(train_vars)
# predicted = model.predict(test_vars, batch_size=batch_size)
denormalized_predicted = de_normalize_dataset(predicted.copy(), DIFF_DEMAND)

true_dates = test_dates
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
# axes = plt.gca()
# axes.set_ylim([0, 1])  # seteo limite en el eje y entre 0 y 1
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
plus_one = denormalized_true_net_demand + 1
error_ds = diff / plus_one
# # error_ds = diff / denormalized_true_net_demand
graph = plot_w_xticks(all_ticks, major_ticks, major_tick_labels, [(error_ds, 'b-o')])
graph.set_title('Error con valores DES-normalizados')
axes = plt.gca()
axes.set_ylim([0, 1])  # seteo limite en el eje y entre 0 y 1
plt.show()

min_min = min(denormalized_true_net_demand.min(), denormalized_predicted_net_demand.min())
a = denormalized_true_net_demand - min_min
b = denormalized_predicted_net_demand - min_min
diff = abs(a - b)
c = a + b
d = diff / c
graph = plot_w_xticks(all_ticks, major_ticks, major_tick_labels, [(d, 'b-o')])
graph.set_title('Error con SUMA')
axes = plt.gca()
axes.set_ylim([0, 1])  # seteo limite en el eje y entre 0 y 1
plt.show()



# PLOTEO LOS VALORES PREDECIDOS Y LOS ORIGINALES DES-NORMALIZADOS (en miles de pesos) -----------------------------------------------------------------------------
graph = plot_w_xticks(all_ticks, major_ticks, major_tick_labels, [(denormalized_true_net_demand / 1000, 'b-o'),
                                                                  (denormalized_predicted_net_demand / 1000, 'r-o')])
graph.set_ylabel('Dinero en MILES')
plt.grid()
plt.show()





plt.plot(DIFF_DEMAND)
# graph = plot_w_xticks(all_ticks, major_ticks, major_tick_labels, [(DIFF_DEMAND,'b')])
plt.show()





graph = plot_w_xticks(all_ticks, major_ticks, major_tick_labels, [(denormalized_predicted_net_demand / 1000, 'r-o')])
graph.set_ylabel('PRONOSTICO en MILES de pesos')
plt.grid()
plt.show()


graph = plot_w_xticks(all_ticks, major_ticks, major_tick_labels, [(denormalized_true_net_demand / 1000, 'r-o')])
graph.set_ylabel('POSTA en MILES de pesos')
plt.grid()
plt.show()
