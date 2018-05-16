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

input_file = 'full_entrada_salida_pesos_506.csv'

# COLUMNAS:
#  0     1   2    3     4         5      6         7          8      9
# index,dow,dom,month,year,in_demand,out_demand,prev_holi,pos_holi,minfl
vars_df = pandas.read_csv(input_file, usecols=[1, 2, 3, 7, 8])
demand_df = pandas.read_csv(input_file, usecols=[5, 6])

#  0     1   2    3     4         5      6         7          8      9
# index,dow,dom,month,year,in_demand,out_demand,prev_holi,pos_holi,minfl
dates_ds = pandas.read_csv(input_file, usecols=[2, 3, 4]).values

DIFF_DEMAND = demand_df.values[:, 0] - demand_df.values[:, 1]
DIFF_DEMAND.resize((len(DIFF_DEMAND), 1))

#  0     1   2    3     4         5      6         7          8      9
# index,dow,dom,month,year,in_demand,out_demand,prev_holi,pos_holi,minfl
dates_ds = pandas.read_csv(input_file, usecols=[2, 3, 4]).values

vars_ds = vars_df.values.astype('float64')
demand_ds = demand_df.values.astype('float64')

in_demand = demand_ds[:, 0]
out_demand = demand_ds[:, 1]
demand_ds = in_demand - out_demand
demand_ds.resize((len(demand_ds), 1))

vars_w_demand = numpy.hstack((vars_ds, demand_ds))

DOW_COL, DOM_COL, MONTH_COL, PREVH_COL, POSTH_COL, NETD_COL = range(6)

# VERIFICO SI EL DATASET TIENE DIAS SALTEADOS
# idx = -1
# prev_dow = None
# for entry in vars_w_demand:
#     idx = idx + 1
#     dow = entry[DOW_COL]
#     if prev_dow is None:
#         prev_dow = dow
#         continue
#     if(prev_dow == 7 and dow == 1):
#         prev_dow = dow
#         continue
#     if(abs(prev_dow - dow) == 1):
#         prev_dow = dow
#         continue
#     print('Error en entrada ' , idx)
#     prev_dow = dow

b = vars_w_demand[:, DOW_COL] == 1
weekstart_idx = b.argmax()
vars_w_demand = vars_w_demand[weekstart_idx:]

b = vars_w_demand[:, DOW_COL] == 7
c = b[::-1] # invierto el arreglo 'b'
weekend_idx = c.argmax() * -1
vars_w_demand = vars_w_demand[:weekend_idx]

week_count = int(len(vars_w_demand) / 7) # cantidad de semanas en el dataset

# obtengo las demandas netas 
weekly_values = []
for week_idx in range(week_count):
    start_idx = 7 * week_idx
    end_idx = start_idx + 7
    weekly_value = numpy.sum(vars_w_demand[start_idx:end_idx, NETD_COL])
    weekly_values.append(weekly_value)

plt.plot(weekly_values)
plt.show()

