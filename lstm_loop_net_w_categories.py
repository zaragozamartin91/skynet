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

DIFF_DEMAND = demand_df.values[:, 0] - demand_df.values[:, 1]
DIFF_DEMAND.resize((len(DIFF_DEMAND), 1))

#  0     1   2    3     4         5      6         7
# index,dow,dom,month,year,in_demand,out_demand,holi_flag
dates_ds = pandas.read_csv(input_file, usecols=[2, 3, 4]).values

demand_ds = demand_df.values.astype('float64')

in_demand = demand_ds[:, 0]
out_demand = demand_ds[:, 1]
demand_ds = in_demand - out_demand
demand_ds.resize((len(demand_ds), 1))

# elimino las entradas con ruido de la primera region
a = noise_remover.remove_max_on_region(demand_ds, 4, 0, 300)
b = noise_remover.remove_min_on_region(a, 5, 0, 300)
demand_ds = b.copy()

# obtengo los valores de demanda categorizados
CAT_COUNT = 40
categorized_demand, demand_bags = categorizer.categorize_real_w_equal_frames(demand_ds.reshape((len(demand_ds), 1)), cat_count=CAT_COUNT)
demand_ds = categorized_demand.copy()

ds_size = len(demand_ds)  # tamano del dataset

# prepare the dataset of input to output pairs encoded as integers
seq_length = 100
dataX = []
dataY = []
for i in range(0, ds_size - seq_length, 1):
    seq_in = demand_ds.flatten()[i:i + seq_length]
    seq_out = demand_ds.flatten()[i + seq_length]
    dataX.append(seq_in)
    dataY.append(seq_out)

n_patterns = len(dataX)
print("Total Patterns: ", n_patterns)

# reshape X to be [samples, time steps, features]
X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
X = normalize(X, CAT_COUNT - 1)
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
model.add(Dropout(0.2))
model.add(Dense(train_y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')
# define the checkpoint
# filepath = "weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
# checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
# callbacks_list = [checkpoint]
# fit the model
# model.fit(train_x, train_y, epochs=20, batch_size=batch_size, callbacks=callbacks_list , shuffle=False)
model.fit(train_x, train_y, epochs=10, batch_size=batch_size, shuffle=False)


predicted = []

# mi primer dato es el comienzo de los datos de prueba
pattern = dataX[test_lower_limit]

for i in range(test_size):
    x = numpy.reshape(pattern, (1, len(pattern), 1))
    x = normalize(x , CAT_COUNT - 1)
    prediction = model.predict(x, verbose=0)
    predicted_category = numpy.argmax(prediction)
    predicted.append(predicted_category)
    pattern = numpy.append(pattern , predicted_category)
    # pattern.append(predicted_category)
    pattern = pattern[1:len(pattern)]

print("\nDone.")

idx = 0
predicted = []

# idx = 0
test_var = test_vars[idx:idx + 1]
predicted_entry = model.predict(test_var, batch_size=batch_size).reshape((1,))
predicted.append(predicted_entry)
idx += 1

for idx in range(1, test_size):
    test_var = test_vars[idx:idx + 1]  # obtengo la entrada sobre la cual voy a hacer la prediccion
    start_ts = 0 if (timesteps - idx + 1 <= 0) else (timesteps - idx + 1)
    end_ts = timesteps + 1 + 1  # si recuerdo 10 dias , entonces tengo 11 elementos (10 pasados + 1 actual)
    start_pred = 0 if (idx - timesteps - 1 <= 0) else idx - timesteps - 1
    end_pred = idx
    test_var[0, start_ts:end_ts, 5:] = predicted[start_pred:end_pred]
    predicted_entry = model.predict(test_var, batch_size=batch_size).reshape((1,))
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
