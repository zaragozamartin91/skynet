import pandas
import numpy as np
from skdate import date_parser

DEFAULT_DELIM = ';'

DATE_IDX=0
DOW_IDX=1
DOM_IDX=2
MONTH_IDX=3
YEAR_IDX=4
IN_DEMAND_IDX = 5
OUT_DEMAND_IDX = 6

DEF_COLS = [ DATE_IDX,DOW_IDX,DOM_IDX,MONTH_IDX,YEAR_IDX,IN_DEMAND_IDX,OUT_DEMAND_IDX ]

DOW_MONDAY = 2
DOW_FRIDAY = 6

def build_dataframe(file_path , delim = DEFAULT_DELIM , cols = DEF_COLS):
  """ Obtiene un dataframe a partir del archivo de dias, entrada y salida """
  return pandas.read_csv(file_path, delimiter=delim, usecols=cols, engine='python')

def fill_dataframe(dataframe):
  dataset = dataframe.values
  dataset_row_count = dataset[:,0].size
  full_dataset = []
  
  idx = 0
  next_idx = idx + 1
  is_last_entry = next_idx == dataset_row_count

  while(True):
    curr_entry = dataset[idx]
    add_entry(curr_entry , full_dataset)
    
    if(is_last_entry): break
    next_entry = dataset[next_idx]

    if(is_missing(curr_entry , next_entry)):
      missing_entry = build_missing_entry(curr_entry)
      curr_entry = missing_entry
      continue

    idx+=1
    next_idx = idx+1
    is_last_entry = next_idx == dataset_row_count

  return np.array( full_dataset )


def build_missing_entry(curr_entry):
  """ Crea la entrada del siguiente dia con valores de demanda de entrada y salida iguales a cero """
  missing_entry = curr_entry.copy()
  datestr = get_date(missing_entry)
  datewr = date_parser.parse_date_wformat(datestr)
  start_date = datewr['full_date']
  end_date = date_parser.add_days(start_date)
  
  set_date(missing_entry , date_parser.to_string(end_date))
  set_din(missing_entry , 0.0)
  set_dout(missing_entry, 0.0)

  return missing_entry


def is_missing(curr_entry , next_entry):
  """ Determina si falta una entrada entre curr_entry y next_entry """
  curr_dow = get_dow(curr_entry)
  next_dow = get_dow(next_entry)
  if (is_friday_to_monday(curr_dow , next_dow)) : return False
  return abs(next_dow - curr_dow) > 1

def is_friday_to_monday(curr_dow , next_dow):
  """ Determina si el salto de dias en la semana es de viernes a lunes """
  return curr_dow == DOW_FRIDAY and next_dow == DOW_MONDAY

def get_dow(entry):
  return entry[DOW_IDX]

def add_entry(entry , dataset):
  """ Agrega un entry de tipo object al dataset """
  # dataset.append( np.array(entry , dtype='object') )
  dataset.append( entry )

def get_date(entry):
  """ Obtiene el campo fecha de un entry del dataset """
  return entry[DATE_IDX]

def set_date(entry , date):
  """ Establece el valor del campo fecha de un entry del dataset """
  entry[DATE_IDX] = date


def set_din(entry , din):
  entry[IN_DEMAND_IDX] = din

def set_dout(entry , dout):
  entry[OUT_DEMAND_IDX] = dout


