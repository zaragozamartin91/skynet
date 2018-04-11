from skmatrix import var_matrix_builder
from skmatrix import out_matrix_builder
import pandas


file_path = 'entrada_salida_pesos_540.csv'

dataframe = out_matrix_builder.build_dataframe(file_path)
print(dataframe)


dataframe = var_matrix_builder.build_dataframe(file_path)
print(dataframe)
