import pandas
import numpy as np

DEFAULT_DELIM = ';'
IN_DEMAND_IDX = 4
OUT_DEMAND_IDX = 5


def build_dataframe(file_path , delim = DEFAULT_DELIM):
  dataframe = pandas.read_csv(file_path, delimiter=delim, usecols=[IN_DEMAND_IDX,OUT_DEMAND_IDX], engine='python')
  return dataframe.values

def write_to_csv(file_path , dataframe):
  dataframe.to_csv(file_path)
