import sys
import pandas
import numpy
from skmatrix import demand_matrix
from skmatrix import full_matrix
""" Parsea un archivo de demandas de entrada y salida y obtiene uno con tados completos"""

MIN_ARG_COUNT = 3
argv = sys.argv
if (len(argv) < MIN_ARG_COUNT):
    print('Argumentos: archivo_entrada archivo_salida [net]')
    sys, exit(1)

demand_file_path = argv[1]
out_file_path = argv[2]
net = False
if (len(argv) > 3): net = bool(argv[3])

demand_df = demand_matrix.build_dataframe(demand_file_path)
demand_ds = demand_matrix.fill_dataset(demand_df.values)
# obtengo el dataset de demadas con dias no laborables
full_demand_ds = demand_matrix.fill_dataset_wholidays(demand_ds)

print(full_demand_ds)

# DATE_IDX        0
# DOW_IDX         1
# DOM_IDX         2
# MONTH_IDX       3
# YEAR_IDX        4
# IN_DEMAND_IDX   5
# OUT_DEMAND_IDX  6
cols = ['dow', 'dom', 'month', 'year', 'in_demand', 'out_demand', 'prev_holy', 'pos_holy']

if(net):
    print('TRABAJANDO CON DEMANDA NETA...')
    cols = ['dow', 'dom', 'month', 'year', 'net_demand', 'prev_holy', 'pos_holy']
    days_ds = full_demand_ds[:,0:5]
    holy_ds = full_demand_ds[:,7:9]
    net_demand = demand_matrix.get_net_demand(full_demand_ds)
    full_demand_ds = numpy.hstack( (days_ds , net_demand , holy_ds) )

full_df = pandas.DataFrame(data=full_demand_ds[:, 1:], columns=cols)
full_df.to_csv(out_file_path)
