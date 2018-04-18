import sys
import pandas
from skmatrix import demand_matrix
from skmatrix import infl_matrix
from skmatrix import full_matrix

MIN_ARG_COUNT = 3
argv = sys.argv
if(len(argv) < MIN_ARG_COUNT): 
  print('Argumentos: archivo_entrada archivo_salida')
  sys,exit(1)

demand_file_path = argv[1]
out_file_path = argv[2]

demand_df = demand_matrix.build_dataframe(demand_file_path)
demand_ds = demand_matrix.fill_dataset(demand_df.values)
# obtengo el dataset de demadas con dias no laborables
full_demand_ds = demand_matrix.fill_dataset_wholidays(demand_ds)

print(full_demand_ds)

# DOW_IDX
# DOM_IDX
# MONTH_IDX
# YEAR_IDX
# IN_DEMAND_IDX
# OUT_DEMAND_IDX
cols = ['dow', 'dom', 'month', 'year', 'in_demand', 'out_demand', 'prev_holy', 'pos_holy']
full_df = pandas.DataFrame(data=full_demand_ds[:, 1:], columns=cols)
full_df.to_csv(out_file_path)
