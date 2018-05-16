import pandas
from skmatrix import demand_matrix
from skmatrix import infl_matrix
from skmatrix import full_matrix

demand_file_path = 'entrada_salida_pesos_540_2014_2017.csv'
demand_df = demand_matrix.build_dataframe(demand_file_path)
demand_ds = demand_matrix.fill_dataset(demand_df.values)
# obtengo el dataset de demadas con dias no laborables
full_demand_ds = demand_matrix.fill_dataset_wholidays(demand_ds)

print(full_demand_ds)

infl_file_path = 'inflation_J2014_M2017.csv'
infl_df = infl_matrix.build_dataframe(infl_file_path)
infl_ds = infl_matrix.parse_dataset(infl_df.values)

full_ds = full_matrix.build_matrix(full_demand_ds, infl_ds)
print(full_ds)

# DOW_IDX
# DOM_IDX
# MONTH_IDX
# YEAR_IDX
# IN_DEMAND_IDX
# OUT_DEMAND_IDX
cols = ['dow', 'dom', 'month', 'year', 'in_demand', 'out_demand', 'prev_holi', 'pos_holi', 'minfl']
full_df = pandas.DataFrame(data=full_ds[:, 1:], columns=cols)
full_df.to_csv('full_data.csv')

vars_df = pandas.read_csv('full_data.csv', usecols=[1, 2, 3, 7, 8, 9])
demand_df = pandas.read_csv('full_data.csv', usecols=[5,6])

vars_df.to_csv('vars.csv')
demand_df.to_csv('demands.csv')
