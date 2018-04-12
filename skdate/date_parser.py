from dateutil.parser import parse
import datetime

STD_DATE_FORMAT = '%b %d %Y %I:%M%p'


def parse_date(strdate):
    """Parsea una fecha y la convierte en un diccionario con informacion util"""
    d = parse(strdate)
    return wrap_date(d)


def parse_date_wformat(strdate, strformat=STD_DATE_FORMAT):
    """Parsea una fecha en un formato dado"""
    d = datetime.datetime.strptime(strdate, strformat)
    return wrap_date(d)


def add_days(start_date, daycount=1):
    """ Agrega una cantidad de dias determinada a una fecha """
    end_date = start_date + datetime.timedelta(days=daycount)
    return end_date


def to_string(date, format=STD_DATE_FORMAT):
    """ Formatea una fecha como un string """
    return date.strftime(format)


def wrap_date(d):
    return {'weekday': d.weekday(), 'day': d.day, 'month': d.month, 'year': d.year, 'full_date': d}
