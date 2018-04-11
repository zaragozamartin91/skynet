from matrix import var_matrix_builder
from matrix import out_matrix_builder
import pandas


file_path = 'entrada_salida_pesos_540.csv'

dataset = out_matrix_builder.build_dataframe(file_path)
print(dataset)


dataset = var_matrix_builder.build_dataframe(file_path)
print(dataset)
