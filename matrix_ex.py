import pandas
from skmatrix import matrix_builder as mb

file_path = 'entrada_salida_pesos_540_2014_2017.csv'

# dataframe = out_matrix_builder.build_dataframe(file_path)
# print(dataframe)

dataframe = mb.build_dataframe(file_path)
# print(dataframe)

ds = mb.fill_dataset(dataframe.values)
print('')
print('ds: ')
print(ds)
print('ds.size: ')
print(ds.size)
print('ds.dtype: ')
print(ds.dtype)
