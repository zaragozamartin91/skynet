import numpy
import matplotlib.pyplot as plt
import pandas
import math

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras import optimizers
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras.layers import Dropout

from datetime import date
from datetime import datetime

from matplotlib.dates import date2num
from matplotlib.dates import num2date

from fractions import gcd

from skmatrix import categorizer
from skmatrix import noise_remover
from skmatrix import normalizer


def normalize_dataset(ds):
    return normalizer.normalize_dataset(ds.copy())


def normalize(ds, max_val=None, final_shape=None):
    t_max = a.max() if max_val is None else max_val
    t_shape = ds.shape if final_shape is None else final_shape
    a = ds / max_val
    return a.reshape(t_shape)


numpy.random.seed(7)

input_file = 'full_entrada_salida_pesos_151.csv'

# COLUMNAS:
#  0     1   2    3     4         5      6         7
# index,dow,dom,month,year,in_demand,out_demand,holi_flag
demand_df = pandas.read_csv(input_file, usecols=[5, 6])
vars_df = pandas.read_csv(input_file, usecols=[1, 2, 3, 7])

DIFF_DEMAND = demand_df.values[:, 0] - demand_df.values[:, 1]
DIFF_DEMAND.resize((len(DIFF_DEMAND), 1))

#  0     1   2    3     4         5      6         7
# index,dow,dom,month,year,in_demand,out_demand,holi_flag
dates_ds = pandas.read_csv(input_file, usecols=[2, 3, 4]).values

demand_ds = demand_df.values.astype('float64')
vars_ds = vars_df.values.astype('float64')

in_demand = demand_ds[:, 0]
out_demand = demand_ds[:, 1]
demand_ds = in_demand - out_demand
demand_ds.resize((len(demand_ds), 1))

vars_cols = vars_ds.shape[1]
demand_cols = demand_ds.shape[1]

# creo el dataset con todas las columnas a usar
dataset = numpy.hstack((demand_ds, vars_ds))

DEMAND_COL_IDX = 0

# elimino las entradas con ruido de la primera region
a = noise_remover.remove_max_on_region(dataset, 4, 0, 300, col=DEMAND_COL_IDX)
b = noise_remover.remove_min_on_region(a, 5, 0, 300, col=DEMAND_COL_IDX)
dataset = b.copy()

# obtengo los valores de demanda categorizados
CAT_COUNT = 40
categorized_dataset, demand_bags = categorizer.categorize_real_w_equal_frames(dataset, cat_count=CAT_COUNT , cat_col=DEMAND_COL_IDX)
dataset = categorized_dataset.copy()

ds_size = len(dataset)  # tamano del dataset
norm_dataset = normalize_dataset(dataset) # dataset normalizado

col_count = dataset.shape[1]
seq_length = 10 # timesteps a recordar
dataX = []
dataY = []
for i in range(0, ds_size - seq_length, 1):
    start_idx = i * col_count
    end_idx = start_idx + seq_length * col_count
    seq_in = norm_dataset.flatten()[start_idx:end_idx]
    seq_out = dataset.flatten()[end_idx]
    dataX.append(seq_in)
    dataY.append(seq_out)

n_patterns = len(dataX)
print("Total Patterns: ", n_patterns)


# reshape X to be [samples, time steps, features]
X = numpy.reshape(dataX, (n_patterns, seq_length, col_count))
# one hot encode the output variable
y = np_utils.to_categorical(dataY)

batch_size = 5
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

# define the LSTM model
model = Sequential()
model.add(LSTM(256, input_shape=(train_x.shape[1], train_x.shape[2])))
# model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dense(train_y.shape[1], activation='softmax'))
opt = optimizers.adam(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer=opt)
# define the checkpoint
# filepath = "weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
# checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
# callbacks_list = [checkpoint]
# fit the model
# model.fit(train_x, train_y, epochs=20, batch_size=batch_size, callbacks=callbacks_list , shuffle=False)
model.fit(train_x, train_y, epochs=200, batch_size=batch_size, shuffle=False)

predicted = []

# mi primer dato es el comienzo de los datos de prueba
pattern = dataX[test_lower_limit]

for i in range(test_size - 1):
    x = numpy.reshape(pattern, (1, seq_length, col_count))
    # x = normalize(x, CAT_COUNT - 1)
    prediction = model.predict(x, verbose=0)
    predicted_category = numpy.argmax(prediction)
    predicted.append(predicted_category)
    norm_predicted_category = predicted_category / (CAT_COUNT - 1)
    next_pattern = dataX[test_lower_limit + i + 1].copy()
    row_to_append = next_pattern[(seq_length-1) * col_count + DEMAND_COL_IDX : ]
    row_to_append[DEMAND_COL_IDX] = norm_predicted_category
    pattern = numpy.append(pattern, row_to_append)
    # pattern.append(predicted_category)
    pattern = pattern[col_count:len(pattern)]


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
