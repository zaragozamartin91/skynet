from skmatrix import demand_matrix
from skmatrix import infl_matrix
import numpy as np


def build_matrix(demand_ds, infl_ds):
    """ Construye la matriz de entrada a partir de los full datasets de demanda e inflacion """
    infl_values = []
    row_count = demand_ds[:, 0].size
    for idx in range(row_count):
        demand_entry = demand_ds[idx]
        month = demand_matrix.get_month(demand_entry)
        year = demand_matrix.get_year(demand_entry)
        infl_entry = infl_matrix.get_entry_by_year_and_month(year, month, infl_ds)
        minfl_value = infl_matrix.get_minfl(infl_entry)
        infl_values.append((minfl_value,))

    return np.hstack((demand_ds, np.array(infl_values)))
