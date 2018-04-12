import pandas
import numpy as np

DEFAULT_DELIM = ','

COL_IDX = 0
# Indices de columnas del dataset
DATE_IDX, COL_IDX = COL_IDX, COL_IDX + 1
MONTH_INFL_IDX, COL_IDX = COL_IDX, COL_IDX + 1
YEAR_INFL_IDX, COL_IDX = COL_IDX, COL_IDX + 1
YEAR_IDX, COL_IDX = COL_IDX, COL_IDX + 1
MONTH_IDX, COL_IDX = COL_IDX, COL_IDX + 1

DEF_COLS = [0, 2, 3]


def build_dataframe(file_path, delim=DEFAULT_DELIM, cols=DEF_COLS):
    """ Obtiene un dataframe a partir del archivo de dias, entrada y salida """
    return pandas.read_csv(file_path, delimiter=delim, usecols=cols, engine='python')


def get_dataset(dataframe):
    return dataframe.values


def parse_dataset(dataset):
    new_dates = split_date_col(dataset)
    dc = np.array(new_dates)
    return np.hstack((dataset, dc))


def split_date_col(dataset):
    rows = []
    for entry in dataset:
        date = get_date(entry)
        syear, smonth = date.split('m')
        year, month = int(syear), int(smonth)
        rows.append((year, month))

    return rows


def get_date(entry):
    return entry[DATE_IDX]


def get_year(entry):
    return entry[YEAR_IDX]


def get_month(entry):
    return entry[MONTH_IDX]


def get_minfl(entry):
    return entry[MONTH_INFL_IDX]


def get_entry_by_year_and_month(year, month, dataset):
    for entry in dataset:
        if (year == get_year(entry) and month == get_month(entry)):
            return entry
    return None
