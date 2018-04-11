import pandas
import numpy as np
from skdate import date_parser

DEFAULT_DELIM = ';'

# Indices de columnas del dataset
DATE_IDX=0
DOW_IDX=1
DOM_IDX=2
MONTH_IDX=3
YEAR_IDX=4
IN_DEMAND_IDX = 5
OUT_DEMAND_IDX = 6

DEF_COLS = [ DATE_IDX,DOW_IDX,DOM_IDX,MONTH_IDX,YEAR_IDX,IN_DEMAND_IDX,OUT_DEMAND_IDX ]

# dias de sql server
DOW_SUNDAY = 1
DOW_MONDAY = 2
DOW_TUESDAY = 3
DOW_WEDNESDAY = 4
DOW_THURSDAY = 5
DOW_FRIDAY = 6
DOW_SATURDAY = 7

# Mapeo entre dias de la semana de 'datetime' y dias de la semana de sql server
DOW_MAP = [2,3,4,5,6,7,1]

def build_dataframe(file_path , delim = DEFAULT_DELIM , cols = DEF_COLS):
  """ Obtiene un dataframe a partir del archivo de dias, entrada y salida """
  return pandas.read_csv(file_path, delimiter=delim, usecols=cols, engine='python')

def fill_dataset(dataset):
  """ Crea un nuevo dataset con las entradas de dias faltantes """
  dataset_row_count = dataset[:,0].size
  full_dataset = []
  
  idx = 0
  curr_entry = dataset[idx]
  next_idx = idx + 1
  is_last_entry = next_idx == dataset_row_count

  while(True):
    print('curr_entry: %s' % str(curr_entry))
    add_entry(curr_entry , full_dataset)
    
    if(is_last_entry): break
    next_entry = dataset[next_idx]

    if(is_missing(curr_entry , next_entry)):
      missing_entry = build_missing_entry(curr_entry)
      curr_entry = missing_entry
      continue

    idx+=1
    curr_entry = dataset[idx]
    next_idx = idx+1
    is_last_entry = next_idx == dataset_row_count

  return np.array( full_dataset )

def get_var_dataset(dataset):
  return dataset[:,DOW_IDX:IN_DEMAND_IDX].astype('int32')

def get_demand_dataset(dataset):
  return dataset[:,IN_DEMAND_IDX:].astype('float32')


def build_missing_entry(curr_entry):
  """ Crea la entrada del siguiente dia con valores de demanda de entrada y salida iguales a cero """
  print('CONSTRUYENDO ENTRADA FALTANTE')

  missing_entry = curr_entry.copy()
  datestr = get_date(missing_entry)
  datewr = date_parser.parse_date_wformat(datestr)
  start_date = datewr['full_date']
  end_date = date_parser.add_days(start_date , daycount=1)
  
  set_date(missing_entry , date_parser.to_string(end_date))
  dow = DOW_MAP[ end_date.weekday() ]
  set_dow(missing_entry , dow)
  set_dom(missing_entry , datewr['day'])
  set_month(missing_entry , datewr['month'])
  set_year(missing_entry , datewr['year'])
  set_din(missing_entry , 0.0)
  set_dout(missing_entry, 0.0)
  
  return missing_entry


def is_missing(curr_entry , next_entry):
  """ Determina si falta una entrada entre curr_entry y next_entry """
  curr_dow = get_dow(curr_entry)
  next_dow = get_dow(next_entry)
  # if (is_friday_to_monday(curr_dow , next_dow)) : return False
  return abs(next_dow - curr_dow) > 1

def is_friday_to_monday(curr_dow , next_dow):
  """ Determina si el salto de dias en la semana es de viernes a lunes """
  return curr_dow == DOW_FRIDAY and next_dow == DOW_MONDAY


def add_entry(entry , dataset):
  """ Agrega un entry de tipo object al dataset """
  # dataset.append( np.array(entry , dtype='object') )
  dataset.append( entry )


def get_holidays(dataset):
  """Obtiene un arreglo con los indices de entradas que corresponden con dias no laborables """
  count = dataset[:,0].size
  holidays = []
  
  for idx in range(count):
    entry = dataset[idx]
    dow = get_dow(entry)
    if(is_weekend(dow)): continue
    dout = get_dout(entry)
    if(no_demand(dout)): holidays.append(idx)

  return holidays

def is_weekend(dow):  return dow == DOW_SATURDAY or dow == DOW_SUNDAY

def no_demand(dout):  return dout < 1.0

def fill_dataset_wpreholidays(dataset):
  count = dataset[:,0].size
  for idx in range(count):
    pholidays = get_pholidays(idx , dataset)

def get_pholidays(idx , dataset):
  if(idx < 0): return 0 # llegue a la primera entrada

  entry = dataset[idx]
  dow = get_dow(entry)
  if(is_weekend(dow)): return get_pholidays(idx - 1 , dataset) # los fines de semana no cuentan
  
  dout = get_dout(entry)
  if(no_demand(dout)): return 1 + get_pholidays(idx - 1 , dataset) # si no hay demanda, cuento un dia no laborable
  else: return 0

# GETTERS DE entry -----------------------------------------------------------------------------------------------------

def get_dow(entry):
  """ Obtiene el campo de dia de la semana de una entrada del dataset """
  return entry[DOW_IDX]

def get_date(entry):
  """ Obtiene el campo fecha de un entry del dataset """
  return entry[DATE_IDX]

def get_dout(entry):
  return entry[OUT_DEMAND_IDX]

# SETTERS DE entry -----------------------------------------------------------------------------------------------------

def set_date(entry , date):
  """ Establece el valor del campo fecha de un entry del dataset """
  entry[DATE_IDX] = date

def set_dow(entry , dow):
  entry[DOW_IDX] = dow

def set_dom(entry , dom):
  entry[DOM_IDX] = dom

def set_month(entry , month):
  entry[MONTH_IDX] = month

def set_year(entry , year):
  entry[YEAR_IDX] = year

def set_din(entry , din):
  entry[IN_DEMAND_IDX] = din

def set_dout(entry , dout):
  entry[OUT_DEMAND_IDX] = dout


