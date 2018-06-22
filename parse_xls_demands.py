import numpy
import pandas
import datetime

suc = '2'

df = pandas.read_csv('SUC_' + suc + '_DEMANDAXLS.csv', delimiter=';', usecols=[0, 1, 2], engine='python')

ds = df.values
ds_len = len(ds)
ds_cols = ds.shape[1]

DATE_COL, CASHD_COL, ATMD_COL = range(ds_cols)

holiday_b = ds[:, CASHD_COL] == 0

final_ds = ds.copy()
# final_ds[holiday_b, ATMD_COL] = 0 # seteo que la demanda de atm en los dias feriados es tambien 0
final_ds = numpy.hstack([final_ds, holiday_b.reshape([ds_len, 1]).astype('int32')])

cols = ['DATE', 'CASHD', 'ATMD', 'HOLIDAY']
full_df = pandas.DataFrame(data=final_ds, columns=cols)
full_df.to_csv('full_caja_Datm_' + suc + '.csv')
