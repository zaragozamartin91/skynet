from dateutil.parser import parse
import datetime

STD_DATE_FORMAT = '%b %d %Y %I:%M%p'

def parse_date(strdate):
  """Parsea una fecha y la convierte en un diccionario con informacion util"""
  d = parse(strdate)
  return wrap_date(d)

def parse_date_wformat(strdate , strformat):
  """Parsea una fecha en un formato dado"""
  d = datetime.datetime.strptime(strdate, strformat)
  return wrap_date(d)

def add_days(start_date , daycount=1):
  end_date = start_date + datetime.timedelta(days=daycount)
  return end_date

def wrap_date(d):
  return {
    'weekday':d.weekday() , 
    'day':d.day,
    'month':d.month,
    'year':d.year,
    'full_date':d}
