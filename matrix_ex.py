from skmatrix import matrix_builder
import pandas


file_path = 'entrada_salida_pesos_540_2014_2017.csv'

# dataframe = out_matrix_builder.build_dataframe(file_path)
# print(dataframe)


dataframe = matrix_builder.build_dataframe(file_path)
# print(dataframe)

full_dataset = matrix_builder.fill_dataset(dataframe.values)
print('')
print('full_dataset: ')
print(full_dataset)
print('full_dataset.size: ')
print(full_dataset.size)
print('full_dataset.dtype: ')
print(full_dataset.dtype)
