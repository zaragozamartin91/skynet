import pandas
import numpy
from skdate import date_parser as dp

def dates_match(data_entry, balance_entry):
    balance_date_str = balance_entry[DATE_IDX]
    balance_date_wr = dp.parse_date(balance_date_str)
    return (balance_date_wr['day'] == data_entry[DOM_IDX] and balance_date_wr['month'] == data_entry[MONTH_IDX]
            and balance_date_wr['year'] == data_entry[YEAR_IDX])

# FIN DE FUNCIONES --------------------------------------------------------------------------------------------

#   0       1
# fecha;saldo_cierre
balance_df = pandas.read_csv('saldo_cierre_540_pesos.csv', usecols=[0, 1], delimiter=";")
balance_ds = balance_df.values

balance_col_count = balance_ds.shape[1]
DATE_IDX, BALANCE_IDX = range(balance_col_count)

#   0    1   2    3     4        5       6         7          8      9
# index,dow,dom,month,year,in_demand,out_demand,prev_holy,pos_holy,minfl
data_df = pandas.read_csv('full_data.csv', usecols=[2, 3, 4, 6])
data_ds = data_df.values

data_col_count = data_ds.shape[1]
DOM_IDX, MONTH_IDX, YEAR_IDX, OUTD_IDX = range(data_col_count)

full_balance_arr = []
balance_idx = 0
start_date_match = False
balance_acc = 0.0
for data_idx in range(len(data_ds)):
    if (balance_idx == len(balance_ds)): break  # si llego al final del dataset de balance entonces termino el ciclo
    data_entry = data_ds[data_idx]
    balance_entry = balance_ds[balance_idx]
    
    date_match = dates_match(data_entry, balance_entry)  # determino si las fechas de las entradas coinciden
    start_date_match = start_date_match or date_match
    if (not start_date_match): continue  # si aun no encontre la fecha de inicio de coincidencia de ambos datasets, entonces continuo
    
    dom = int(data_entry[DOM_IDX])
    month = int(data_entry[MONTH_IDX])
    year = int(data_entry[YEAR_IDX])

    if (date_match):
        balance_value = balance_entry[BALANCE_IDX]
        full_balance_arr.append((dom, month, year, balance_value))
        balance_idx += 1
        balance_acc += balance_value
    else:
        balance_mean = balance_acc / balance_idx
        full_balance_arr.append((dom, month, year, balance_mean))

full_balance_ds = numpy.array(full_balance_arr , dtype='object')
cols = ['dom', 'month', 'year', 'balance']
full_balance_df = pandas.DataFrame(data=full_balance_ds, columns=cols)
full_balance_df.to_csv('full_balance.csv')

