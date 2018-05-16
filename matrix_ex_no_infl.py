import sys
import pandas
import numpy
from skmatrix import demand_matrix
from skmatrix import full_matrix
""" Parsea un archivo de demandas de entrada y salida y obtiene uno con tados completos"""

MIN_ARG_COUNT = 5
argv = sys.argv
if (len(argv) < MIN_ARG_COUNT):
    print('Argumentos: archivo_entrada archivo_salida net|out holi_flag|holi_count')
    sys, exit(1)

demand_file_path = argv[1]
out_file_path = argv[2]

demand_df = demand_matrix.build_dataframe(demand_file_path)
demand_ds = demand_matrix.fill_dataset(demand_df.values) # agrego las entradas faltantes

# columnas finales del dataset
cols = ['dow', 'dom', 'month', 'year']

ds_size = len(demand_ds)

net = argv[3] == 'net'
if net:
    net_demand = demand_ds[:,-2] - demand_ds[:,-1]
    demand_ds = numpy.hstack( ( demand_ds[:,0:-2] , net_demand.reshape((ds_size,1)) ) )
    cols.append('net_demand')
else:
    cols.append('in_demand')
    cols.append('out_demand')



holi_flag = argv[4] == 'holi_flag'
if holi_flag:
    b = abs(demand_ds[:,-1]) < 1.0
    b = b.astype('int32')
    demand_ds = numpy.hstack((demand_ds , b.reshape((ds_size,1))))
    cols.append('holiday')
else:
    # obtengo el dataset de demadas con dias no laborables
    demand_ds = demand_matrix.fill_dataset_wholidays(demand_ds)
    cols.append('prev_holi')
    cols.append('pos_holi')


full_df = pandas.DataFrame(data=demand_ds[:, 1:], columns=cols)
full_df.to_csv(out_file_path)
