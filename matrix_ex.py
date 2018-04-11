from skmatrix import var_matrix_builder
from skmatrix import out_matrix_builder
import pandas


file_path = 'entrada_salida_pesos_540.csv'

# dataframe = out_matrix_builder.build_dataframe(file_path)
# print(dataframe)


dataframe = var_matrix_builder.build_dataframe(file_path)
# print(dataframe)

full_dataframe = var_matrix_builder.fill_dataset(dataframe.values)
print('')
print('full_dataframe: ')
print(full_dataframe)
print('full_dataframe.size: ')
print(full_dataframe.size)
print('full_dataframe.dtype: ')
print(full_dataframe.dtype)
