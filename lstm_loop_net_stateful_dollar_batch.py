import numpy
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras import optimizers
from keras.utils import np_utils
from keras.preprocessing.sequence import TimeseriesGenerator

from datetime import date
from datetime import datetime

from matplotlib.dates import date2num
from matplotlib.dates import num2date

from fractions import gcd

from skmatrix import normalizer
from skmatrix import categorizer
from skmatrix import noise_remover


def normalize_dataset(dataset):
    return normalizer.normalize_dataset(dataset.copy())


def de_normalize_dataset(normalized_ds, original_ds):
    return normalizer.de_normalize_dataset(normalized_ds.copy(), original_ds)


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
    __vars_ds = vars_ds.copy()[1:]
    __demand_ds = demand_ds[:-1]
    return numpy.hstack([__vars_ds, __demand_ds])


def append_curr_demand(vars_ds, demand_ds):
    return numpy.hstack([vars_ds, demand_ds])


numpy.random.seed(7)

input_file = 'full_entrada_salida_pesos_151.csv'
dollar_file = 'dollar_stats_ord.csv'

# COLUMNAS:
#  0     1   2    3     4       5       6         7
# index,dow,dom,month,year,in_demand,out_demand,holiday
vars_df = pandas.read_csv(input_file, usecols=[1, 2, 3, 7])
demand_df = pandas.read_csv(input_file, usecols=[5, 6])

#  0     1   2    3     4       5       6         7
# index,dow,dom,month,year,in_demand,out_demand,holiday
dates_ds = pandas.read_csv(input_file, usecols=[2, 3, 4]).values

dollar_ds = pandas.read_csv(dollar_file, usecols=[5]).values
# a = dollar_ds[:1] - dollar_ds[:-1]
# dollar_ds = numpy.vstack( ([0] , a) )

vars_ds = vars_df.values.astype('float64')
demand_ds = demand_df.values.astype('float64')

# guardo una copia del dataset de demanda completo (entrada y salida)
WHOLE_DEMAND = demand_ds.copy()

DEMAND_TYPE = 'in'

# prueba usando la demanda NETA
if DEMAND_TYPE == 'net':
    in_demand = demand_ds[:, 0]
    out_demand = demand_ds[:, 1]
    demand_ds = in_demand - out_demand
    demand_ds.resize((len(demand_ds), 1))

# prueba usando la demanda DE SALIDA
if DEMAND_TYPE == 'out':
    out_demand = demand_ds[:, 1]
    demand_ds = out_demand
    demand_ds.resize((len(demand_ds), 1))

# prueba usando la demanda DE ENTRADA
if DEMAND_TYPE == 'in':
    in_demand = demand_ds[:, 0]
    demand_ds = in_demand
    demand_ds.resize((len(demand_ds), 1))

vars_ds = numpy.hstack([vars_ds, dollar_ds])
VARS_COL_COUNT = vars_ds.shape[1] # guardo la cantidad de columnas del dataset de variables original

a = numpy.hstack([demand_ds, vars_ds , WHOLE_DEMAND])
b = noise_remover.remove_max(a, 1)
c = noise_remover.remove_min(b, 1)

demand_ds = c[:, 0:1]
vars_ds = c[:, 1:VARS_COL_COUNT+1]
WHOLE_DEMAND = c[:,VARS_COL_COUNT+1:]

# GUARDO los valores originales de demanda para calcular el error mas adelante
DEMAND = demand_ds.copy()

CAT_COUNT = 50
demand_ds, DEMAND_CATEGORIES = categorizer.categorize_real_w_equal_frames(demand_ds, CAT_COUNT, cat_col=0)

# Agrego los valores de la demanda del dia anterior
vars_ds = append_prev_demand(vars_ds, demand_ds)
# Descarto el primer dia dado que no cuenta con demanda previa o su demanda previa es un falso 0.0
demand_ds = demand_ds[1:]
DEMAND = DEMAND[1:]
dates_ds = dates_ds[1:]
WHOLE_DEMAND = WHOLE_DEMAND[1:]

vds_col_count = vars_ds.shape[1]
DOW_COL, DOM_COL, MONTH_COL, HOLIDAY_COL, DOLLAR_COL, DEMAND_COL = range(vds_col_count)

# Asigno valores de 0 a 1 a todas las entradas
norm_vars_ds = normalize_dataset(vars_ds)
norm_demand_ds = normalize_dataset(demand_ds)

# CONSTRUYO LA MATRIZ DE VENTANA
# si el LSTM a usar es stateful=True entones batch_size deberia ser 1
batch_size = 1
seq_length = batch_size  # timesteps a recordar
vds_size = len(vars_ds)
dataX = []
dataY = []
for i in range(0, vds_size - seq_length + 1, 1):
    start_idx = i * vds_col_count
    end_idx = start_idx + seq_length * vds_col_count
    seq_in = norm_vars_ds.flatten()[start_idx:end_idx]
    seq_out = demand_ds.flatten()[i + seq_length - 1]
    dataX.append(seq_in)
    dataY.append(seq_out)

n_patterns = len(dataX)
print("Total Patterns: ", n_patterns)
# sincronizo los valores de fechas y la demanda original
dates_ds = dates_ds[seq_length - 1:]
# DEMAND = DEMAND[:-seq_length]
DEMAND = DEMAND[seq_length - 1:]
WHOLE_DEMAND = WHOLE_DEMAND[seq_length - 1:]

# reshape X to be [samples, time steps, features]
X = numpy.reshape(dataX, (n_patterns, seq_length, vds_col_count))
# one hot encode the output variable
y = np_utils.to_categorical(dataY)

test_size = 30
train_size = batch_size * int((len(X) - test_size) / batch_size)

train_lower_limit = 0
train_upper_limit = train_size
test_lower_limit = train_size
test_upper_limit = train_size + test_size

train_x = X[train_lower_limit:train_upper_limit]
train_y = y[train_lower_limit:train_upper_limit]
test_x = X[test_lower_limit:test_upper_limit]
test_y = y[test_lower_limit:test_upper_limit]
true_dates = dates_ds[test_lower_limit:test_upper_limit]  # obtengo las fechas a graficar

model = Sequential()

epochs = 100
model.add(LSTM(train_y.shape[1] * 4, stateful=True, batch_input_shape=(batch_size, train_x.shape[1], train_x.shape[2])))

model.add(Dense(train_y.shape[1] * 2))
model.add(Dense(train_y.shape[1], activation='softmax'))
opt = optimizers.adam(lr=0.001)
# model.compile(loss='binary_crossentropy', optimizer=opt)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.compile(loss='mean_squared_error', optimizer=opt)
model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, shuffle=False, verbose=1)

predicted = []

# mi primer dato es el comienzo de los datos de prueba
pattern = dataX[test_lower_limit]

last_idx = test_size - 1
for i in range(test_size):
    a = pattern.reshape([1, vds_col_count * seq_length])
    b = numpy.repeat(a, batch_size, axis=0)
    # obtenemos la prediccion del dia N
    x = numpy.reshape(b, (batch_size, seq_length, vds_col_count))
    prediction = model.predict(x, verbose=0)
    predicted_category = numpy.argmax(prediction[0])
    predicted.append(predicted_category)
    if i == last_idx:
        break
    # normalizo el valor de la prediccion
    norm_predicted_category = predicted_category / (CAT_COUNT - 1)
    # luego obtenemos los datos de entrada de la prediccion N+1 ...
    next_pattern = dataX[test_lower_limit + i + 1].copy()
    pattern_to_append = next_pattern[(seq_length - 1) * vds_col_count:]
    # ... y modificamos el campo de demanda del dia anterior para que tenga el valor obtenido en la prediccion
    pattern_to_append[DEMAND_COL] = norm_predicted_category
    pattern = numpy.append(pattern, pattern_to_append)
    # desplazamos el patron un frame
    pattern = pattern[vds_col_count:]

predicted = numpy.array(predicted)

# Obtenemos los ticks para graficar usando fechas...
last_date_idx = test_size - 1
start_date = date.today().replace(true_dates[0, 2], true_dates[0, 1], true_dates[0, 0])  # obtengo la fecha de inicio
end_date = date.today().replace(true_dates[last_date_idx, 2], true_dates[last_date_idx, 1], true_dates[last_date_idx, 0])  # obtengo la fecha de fin
all_ticks = numpy.linspace(date2num(start_date), date2num(end_date), test_size)  # obtengo un arreglo con todos los valores numericos de fechas

tick_spacing = test_size if test_size <= 60 else 12
date_format = "%m/%d" if test_size <= 60 else "%y/%m"

major_ticks = numpy.linspace(date2num(start_date), date2num(end_date), tick_spacing)  # obtengo un arreglo con los valores de fecha que quiero mostrar
major_tick_labels = [date.strftime(date_format) for date in num2date(major_ticks)]

# PLOTEO de las categorias predecidas vs las reales
true_net_demand = dataY[test_lower_limit:test_upper_limit]
predicted_net_demand = predicted
plot_w_xticks(all_ticks, major_ticks, major_tick_labels, [(true_net_demand, 'b-o'), (predicted_net_demand, 'r-o')])
axes = plt.gca()
#axes.set_ylim([20, 40])  # seteo limite en el eje y entre 0 y 1
plt.show()

# MIDO EL ERROR CATEGORICO
CAT_FRAME_SIZE = (DEMAND_CATEGORIES[1] - DEMAND_CATEGORIES[0])[0]  # tamano de franja de categoria
demand_delta = abs(numpy.array(true_net_demand) - predicted)  # diferencias de categoria
c = DEMAND.copy()
d = c - c.min()
d = d[test_lower_limit:test_upper_limit]  # demanda original en valores positivos
error = demand_delta * CAT_FRAME_SIZE / (d.flatten() + 1)  # obtengo el error punto a punto
plt.plot(error, 'r-o')
axes = plt.gca()
axes.set_ylim([0, 1])  # seteo limite en el eje y entre 0 y 1
plt.show()

# Escribo el dinero predecido en un archivo
predicted_money = predicted * CAT_FRAME_SIZE
predicted_money_df = pandas.DataFrame(data=predicted_money, columns=[DEMAND_TYPE])
predicted_money_df.to_csv(DEMAND_TYPE + '.csv')

# MIDO EL ERROR CATEGORICO EN EL PEOR ESCENARIO
# DEFINO EL PEOR ESCENARIO COMO AQUEL EN EL QUE LA DIFERENCIA DE UNA CATEGORIA ES EN REALIDAD DE 2 FRANJAS (MAXIMA DIFERENCIA)
# ASUMO QUE LOS VALORES 0 (NO LABORABLES) FUERON PREDECIDOS CORRECTAMENTE
b = demand_delta > 0
dd = demand_delta.copy()
dd[b] = dd[b] + 1
error = dd * CAT_FRAME_SIZE / (d.flatten() + 1)  # obtengo el error punto a punto
plt.plot(error, 'r-o')
axes = plt.gca()
axes.set_ylim([0, 1])  # seteo limite en el eje y entre 0 y 1
plt.show()

# MIDO EL ERROR EN DINERO REAL
c = DEMAND.copy()
d = c - c.min()
d = d[test_lower_limit:test_upper_limit]  # demanda original en valores positivos
e = numpy.array(DEMAND_CATEGORIES)[predicted] - c.min() - CAT_FRAME_SIZE
error = abs(d - e) / (d + 1)
plt.plot(error, 'r-o')
axes = plt.gca()
axes.set_ylim([0, 1])  # seteo limite en el eje y entre 0 y 1
plt.show()
