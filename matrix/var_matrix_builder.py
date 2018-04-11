import pandas
import numpy as np

DEFAULT_DELIM = ';'

DATE_IDX=0
DOW_IDX=1
DOM_IDX=2
MONTH_IDX=3
YEAR_IDX=4

DEF_COLS = [ DATE_IDX,DOW_IDX,DOM_IDX,MONTH_IDX,YEAR_IDX ]

def build_dataframe(file_path , delim = DEFAULT_DELIM , cols = DEF_COLS):
  """ Obtiene un dataframe a partir del archivo de dias, entrada y salida """
  return pandas.read_csv(file_path, delimiter=delim, usecols=cols, engine='python')

def get_holidays_by_dow(dataframe):
  """ Obtiene un arreglo con los indices de los dias no laborables del dataframe a partir de los dias de semana """
  # obtengo un 
  dataset = dataframe.values[:,1].ravel().astype('int32')
  idx = 0

