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


numpy.random.seed(7)

input_file = 'full_entrada_salida_pesos_151.csv'

# COLUMNAS:
#  0     1   2    3     4       5       6         7
# index,dow,dom,month,year,in_demand,out_demand,holiday
vars_df = pandas.read_csv(input_file, usecols=[1, 2, 3, 7])
demand_df = pandas.read_csv(input_file, usecols=[5, 6])

DIFF_DEMAND = demand_df.values[:, 0] - demand_df.values[:, 1]
DIFF_DEMAND.resize((len(DIFF_DEMAND), 1))

#  0     1   2    3     4       5       6         7
# index,dow,dom,month,year,in_demand,out_demand,holiday
dates_ds = pandas.read_csv(input_file, usecols=[2, 3, 4]).values

vars_ds = vars_df.values.astype('float64')
demand_ds = demand_df.values.astype('float64')

in_demand = demand_ds[:, 0]
out_demand = demand_ds[:, 1]
demand_ds = in_demand - out_demand
demand_ds.resize((len(demand_ds), 1))

CAT_COUNT = 100
demand_ds, DEMAND_CATEGORIES = categorizer.categorize_real_w_equal_frames(demand_ds, CAT_COUNT, cat_col=0)

# Agrego los valores de la demanda del dia anterior
vars_ds = append_prev_demand(vars_ds, demand_ds)
# Descarto el primer dia dado que no cuenta con demanda previa o su demanda previa es un falso 0.0
demand_ds = demand_ds[1:]
dates_ds = dates_ds[1:]

vds_col_count = vars_ds.shape[1]
DOW_COL, DOM_COL, MONTH_COL, HOLIDAY_COL, DEMAND_COL = range(vds_col_count)

# Asigno valores de 0 a 1 a todas las entradas
norm_vars_ds = normalize_dataset(vars_ds)
norm_demand_ds = normalize_dataset(demand_ds)

# CONSTRUYO LA MATRIZ DE VENTANA
# si el LSTM a usar es stateful=True entones batch_size deberia ser 1
batch_size = 1
seq_length = 1  # timesteps a recordar
vds_size = len(norm_vars_ds)
dataX = []
dataY = []
for i in range(0, vds_size - seq_length, 1):
    start_idx = i * vds_col_count
    end_idx = start_idx + seq_length * vds_col_count
    seq_in = norm_vars_ds.flatten()[start_idx:end_idx]
    seq_out = demand_ds.flatten()[i]
    dataX.append(seq_in)
    dataY.append(seq_out)

n_patterns = len(dataX)
print("Total Patterns: ", n_patterns)
dates_ds = dates_ds[1:]  # sincronizo nuevamente el ds de fechas

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
model.add(LSTM(60, stateful=True, batch_input_shape=(batch_size, train_x.shape[1], train_x.shape[2])))
# NO SE COMO HACER ANDAR ESTA CAPA
#model.add(LSTM(5, input_shape=(train_x.shape[1], train_x.shape[2])))
model.add(Dense(30))
model.add(Dense(train_y.shape[1], activation='softmax'))
opt = optimizers.adam(lr=0.001)
model.compile(loss='binary_crossentropy', optimizer=opt)
# model.compile(loss='mean_squared_error', optimizer=opt)
model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, shuffle=False, verbose=1)

predicted = []

# mi primer dato es el comienzo de los datos de prueba
pattern = dataX[test_lower_limit]

last_idx = test_size - 1
for i in range(test_size):
    # obtenemos la prediccion del dia N
    x = numpy.reshape(pattern, (1, seq_length, vds_col_count))
    prediction = model.predict(x, verbose=0)
    predicted_category = numpy.argmax(prediction)
    predicted.append(predicted_category)
    if i == last_idx: break
    # normalizo el valor de la prediccion
    norm_predicted_category = predicted_category / (CAT_COUNT - 1)
    # luego obtenemos los datos de entrada de la prediccion N+1 ...
    next_pattern = dataX[test_lower_limit + i + 1].copy()
    row_to_append = next_pattern[(seq_length - 1) * vds_col_count:]
    # ... y modificamos el campo de demanda del dia anterior para que tenga el valor obtenido en la prediccion
    row_to_append[DEMAND_COL] = norm_predicted_category
    pattern = numpy.append(pattern, row_to_append)
    # desplazamos el patron un frame
    pattern = pattern[vds_col_count:len(pattern)]

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
true_net_demand = dataY[test_lower_limit:]
predicted_net_demand = predicted
plot_w_xticks(all_ticks, major_ticks, major_tick_labels, [(true_net_demand, 'b-o'), (predicted_net_demand, 'r-o')])
plt.show()
