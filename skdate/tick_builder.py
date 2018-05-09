import numpy
from datetime import date
from matplotlib.dates import date2num
from matplotlib.dates import num2date

# Obtiene una tupla con los ticks , major ticks y tick labels a partir de un conjunto de fechas
# El conjunto de fechas debe ser un dataset con tuplas: dia , mes , anio.
def build_ticks(test_dates):
  test_size = len(test_dates)
  true_dates = test_dates
  last_date_idx = test_size - 1
  start_date = date.today().replace(true_dates[0, 2], true_dates[0, 1], true_dates[0, 0])  # obtengo la fecha de inicio
  end_date = date.today().replace(true_dates[last_date_idx, 2], true_dates[last_date_idx, 1], true_dates[last_date_idx, 0])  # obtengo la fecha de fin
  all_ticks = numpy.linspace(date2num(start_date), date2num(end_date), test_size)  # obtengo un arreglo con todos los valores numericos de fechas

  tick_spacing = test_size if test_size <= 60 else 30
  date_format = "%m/%d" if test_size <= 60 else "%y/%m"

  # major_ticks = numpy.arange(date2num(start_date), date2num(end_date), tick_spacing)  # obtengo un arreglo con los valores de fecha que quiero mostrar
  major_ticks = numpy.linspace(date2num(start_date), date2num(end_date), tick_spacing)  # obtengo un arreglo con los valores de fecha que quiero mostrar
  major_tick_labels = [date.strftime(date_format) for date in num2date(major_ticks)]

  return all_ticks, major_ticks, major_tick_labels
