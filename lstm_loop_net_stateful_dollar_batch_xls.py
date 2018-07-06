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
from keras import initializers

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


def plot_w_xticks(all_xticks, major_xticks, major_xticks_labels, yplots, ylabels=None):
    """ 
    Plotea sets de datos
    :param all_xticks : todos los valores del eje x
    :param major_xticks : los valores principales del eje x (valores a mostrar)
    :param major_xticks_labels : labels de los principales valores del eje x (arreglo de strings)
    :param yplots : arreglo de tuplas '(datos_eje_y , color_ploteo)'
    :param ylabels : [OPCIONAL] nombres de las curvas
    """
    fig = plt.figure()
    graph = fig.add_subplot(111)
    x = all_xticks
    for idx in range(len(yplots)):
        yplot = yplots[idx]
        if ylabels is None:
            graph.plot(x, yplot[0], yplot[1])
        else:
            graph.plot(x, yplot[0], yplot[1], label=ylabels[idx])
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


def measure_accuracy(true_money, predicted_money, cat_frame_size):
    a = abs(true_money - predicted_money) < (cat_frame_size * 2)
    true_count = a.astype('int32').sum()
    return true_count / a.shape[0]


numpy.random.seed(7)

# CONFIGURACION -----------------------------------------------------------------------------------------------------------------------

suc = '2'  # numero de sucursal
DEMAND_TYPE = 'cash'  # tipo de demanda a medir
CAT_COUNT = 50  # cantidad de categorias de dinero
batch_size = 1  # batch de entrenamiento
seq_length = 10  # timesteps a recordar
test_size = 31
epochs = 175
input_file = 'full_caja_Datm_' + suc + '.csv'

# ------------------------------------------------------------------------------------------------------------------------------------

# COLUMNAS:
#  0     1     2    3     4
# index,DATE,CASHD,ATMD,HOLIDAY
vars_df = pandas.read_csv(input_file, usecols=[4])
demand_df = pandas.read_csv(input_file, usecols=[2, 3])

#  0     1     2    3     4
# index,DATE,CASHD,ATMD,HOLIDAY
dates_ds = pandas.read_csv(input_file, usecols=[1]).values

vars_ds = vars_df.values.astype('float64')
demand_ds = demand_df.values.astype('float64')

# guardo una copia del dataset de demanda completo
WHOLE_DEMAND = demand_ds.copy()

# prueba usando la demanda DE CAJA
if DEMAND_TYPE == 'cash':
    demand_ds = demand_ds[:, 0]
    demand_ds.resize((len(demand_ds), 1))

# prueba usando la demanda DE ATM
if DEMAND_TYPE == 'atm':
    demand_ds = demand_ds[:, 1]
    demand_ds.resize((len(demand_ds), 1))

# prueba usand la suma de las demandas
if DEMAND_TYPE == 'all':
    demand_ds = demand_ds[:, 0] + demand_ds[:, 1]
    demand_ds.resize((len(demand_ds), 1))

VARS_COL_COUNT = vars_ds.shape[1]  # guardo la cantidad de columnas del dataset de variables original

# GUARDO los valores originales de demanda para calcular el error mas adelante
DEMAND = demand_ds.copy()
MINV = DEMAND.min()

demand_ds, DEMAND_CATEGORIES = categorizer.categorize_real_w_equal_frames(demand_ds, CAT_COUNT, cat_col=0)
DEMAND_CATEGORIES = numpy.array(DEMAND_CATEGORIES).reshape([CAT_COUNT, 1])

# Agrego los valores de la demanda del dia anterior
vars_ds = append_prev_demand(vars_ds, demand_ds)
# Descarto el primer dia dado que no cuenta con demanda previa o su demanda previa es un falso 0.0
demand_ds = demand_ds[1:]
DEMAND = DEMAND[1:]
dates_ds = dates_ds[1:]
WHOLE_DEMAND = WHOLE_DEMAND[1:]

vds_col_count = vars_ds.shape[1]
HOLIDAY_COL, DEMAND_COL = range(vds_col_count)

# Asigno valores de 0 a 1 a todas las entradas
norm_vars_ds = normalize_dataset(vars_ds)
norm_demand_ds = normalize_dataset(demand_ds)

# CONSTRUYO LA MATRIZ DE VENTANA
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

model.add(LSTM(train_y.shape[1] * 4, stateful=True, kernel_initializer=initializers.random_normal(seed=7) ,batch_input_shape=(batch_size, train_x.shape[1], train_x.shape[2])))
model.add(Dense(train_y.shape[1] * 4))
model.add(Dense(train_y.shape[1], activation='softmax'))
opt = optimizers.adam(lr=0.005)
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
start_date = datetime.strptime(true_dates[0, 0], '%Y-%m-%d')  # obtengo la fecha de inicio
end_date = datetime.strptime(true_dates[-1, 0], '%Y-%m-%d')  # obtengo la fecha de fin
all_ticks = numpy.linspace(date2num(start_date), date2num(end_date), test_size)  # obtengo un arreglo con todos los valores numericos de fechas

tick_spacing = test_size if test_size <= 60 else 12
date_format = "%m/%d" if test_size <= 60 else "%y/%m"

major_ticks = numpy.linspace(date2num(start_date), date2num(end_date), tick_spacing)  # obtengo un arreglo con los valores de fecha que quiero mostrar
major_tick_labels = [date.strftime(date_format) for date in num2date(major_ticks)]

# PLOTEO de las categorias predecidas vs las reales
true_net_demand = dataY[test_lower_limit:test_upper_limit]
predicted_net_demand = predicted
plot_w_xticks(all_ticks, major_ticks, major_tick_labels, [(true_net_demand, 'b-o'), (predicted_net_demand, 'r-o')],
              ['categorias reales', 'categorias predecidas'])
axes = plt.gca()
axes.set_ylim([0, 45])  # seteo limite en el eje y entre 0 y 1
plt.legend()
plt.show()

# MIDO EL ERROR CATEGORICO
CAT_FRAME_SIZE = (DEMAND_CATEGORIES[1] - DEMAND_CATEGORIES[0])[0]  # tamano de franja de categoria
demand_delta = abs(numpy.array(true_net_demand) - predicted)  # diferencias de categoria
c = DEMAND.copy()
d = c - c.min()
d = d[test_lower_limit:test_upper_limit]  # demanda original en valores positivos
error = demand_delta * CAT_FRAME_SIZE / (d.flatten() + 1)  # obtengo el error punto a punto
plt.plot(error, 'r-o', label='error de categorias')
axes = plt.gca()
axes.set_ylim([0, 1])  # seteo limite en el eje y entre 0 y 1
plt.legend()
plt.show()

error = abs(numpy.array(true_net_demand) - predicted) / numpy.array(true_net_demand)
plt.plot(error, 'r-o', label='Error entre categorias')
axes = plt.gca()
axes.set_ylim([0, 1])  # seteo limite en el eje y entre 0 y 1
plt.legend()
plt.show()

# MIDO EL ERROR EN DINERO REAL
c = DEMAND.copy()
d = c - c.min()
d = d[test_lower_limit:test_upper_limit]  # demanda original en valores positivos
e = numpy.array(DEMAND_CATEGORIES)[predicted] - c.min() - CAT_FRAME_SIZE
error = abs(d - e) / (d + 1)
plt.plot(error, 'r-o', label='Error de dinero real')
axes = plt.gca()
axes.set_ylim([0, 1])  # seteo limite en el eje y entre 0 y 1
plt.legend()
plt.show()

true_money = DEMAND[test_lower_limit:test_upper_limit]
half_cat_size = CAT_FRAME_SIZE / 2.0
# predicted_money = numpy.array(DEMAND_CATEGORIES)[predicted] - half_cat_size
predicted_money = numpy.array(DEMAND_CATEGORIES)[predicted] - CAT_FRAME_SIZE
plt.plot(true_money, 'b-o', label='Demanda real de dinero')
plt.plot(predicted_money, 'r-o', label='Demanda predecida de dinero')
plt.legend()
plt.show()

# TRABAJANDO CON VALORES DE COE --------------------------------------------------------------------------

# estos son los dias de diciembre que coe utilizo como benchmark
ii = [1, 4, 5, 6, 11, 12, 13, 14, 15, 18, 19, 20, 21, 22, 26, 27, 28]
indexes = numpy.array(ii)
indexes = indexes - 1
coe_predicted_money = predicted_money[indexes]
coe_true_money = true_money[indexes]
at = all_ticks[indexes]
mjt = major_ticks[indexes]
mjtl = numpy.array(major_tick_labels)[indexes]
plot_w_xticks(at, mjt, mjtl, [(coe_true_money, 'b-o'), (coe_predicted_money, 'r-o')], ['dinero real (periodo COE)', 'dinero predecido (periodo COE)'])
plt.legend()
plt.show()

coe_diff = coe_true_money - coe_predicted_money
err = abs(coe_diff / coe_true_money)
plot_w_xticks(at, mjt, mjtl, [(err, 'r-o')], ['Error de dinero en coe'])
axes = plt.gca()
axes.set_ylim([0, 1])  # seteo limite en el eje y entre 0 y 1
plt.legend()
plt.show()



# los valores de COE en el excel estan invertidos
COE_VALUES = {}
COE_VALUES['1'] = numpy.array([
    -5013000, 4013800, 5573300, 7143000, 5148900, 9725700, 5771500, 4166800, 3279400, 6006600, 3652400, 6644600, -1060700, -1301400, 5102700, 4734400, -2921000
]) * -1
COE_VALUES['2'] = numpy.array([
    314500, 1737900, 6216000, 6803600, 1726400, 6531800, 4805600, 2435800, 1764900, 1764900, 3004500, -412000, -18500, -1992400, -2326700, 1515400, 400600
]) * -1

COE_CATEGORIES = categorizer.categorize_arr(COE_VALUES[suc], CAT_FRAME_SIZE, MINV)
predicted_on_coe_dates = numpy.array(predicted)[indexes]
true_on_coe_dates = numpy.array(true_net_demand)[indexes]

coe_diff = abs(COE_CATEGORIES - true_on_coe_dates.flatten())
rrnn_diff = abs(predicted_on_coe_dates - true_on_coe_dates.flatten())

plot_w_xticks(at, mjt, mjtl, [(true_on_coe_dates, 'b-o'), (predicted_on_coe_dates, 'r-o')],
              ['categorias reales (periodo COE)', 'categorias predecidas (periodo COE)'])
plt.show()
