import pandas as pd
import numpy as np
from dateutil.parser import parse
import datetime
from skdate import date_parser

DATE_FORMAT = '%d/%m/%Y'


def fill_dataset(dataset):
    """ Crea un nuevo dataset con las entradas de dias faltantes """
    dataset_row_count = len(dataset)
    full_dataset = []

    idx = 0
    curr_entry = dataset[idx]
    next_idx = idx + 1
    is_last_entry = next_idx == dataset_row_count

    while (True):
        print('curr_entry: %s' % str(curr_entry))
        full_dataset.append(transform_entry(curr_entry))

        if (is_last_entry):
            break
        next_entry = dataset[next_idx]

        if (is_missing(curr_entry, next_entry)):
            missing_entry = build_missing_entry(curr_entry)
            curr_entry = missing_entry
            continue

        idx += 1
        curr_entry = dataset[idx]
        next_idx = idx + 1
        is_last_entry = next_idx == dataset_row_count

    return np.array(full_dataset)


def transform_entry(entry):
    raw_date = get_date(entry)
    date = date_parser.parse_date_wformat(raw_date , DATE_FORMAT)['full_date']
    day, month, year = date.day, date.month, date.year
    value = get_value(entry)
    return np.array([raw_date, int(day), int(month), int(year), float(value)] , 'object')


def is_missing(curr_entry, next_entry):
    """ Determina si falta una entrada entre curr_entry y next_entry """
    curr_date = date_parser.parse_date_wformat(get_date(curr_entry),DATE_FORMAT)
    next_date = date_parser.parse_date_wformat(get_date(next_entry),DATE_FORMAT)
    return date_parser.two_day_gap(curr_date['full_date'], next_date['full_date'])


def build_missing_entry(curr_entry):
    """ Crea la entrada del siguiente dia """
    print('CONSTRUYENDO ENTRADA FALTANTE')
    datestr = get_date(curr_entry)
    datewr = date_parser.parse_date_wformat(datestr, DATE_FORMAT)
    start_date = datewr['full_date']
    end_date = date_parser.add_days(start_date, daycount=1)
    str_end_date = date_parser.to_string(end_date,format='%d/%m/%Y')
    return np.array([str_end_date , get_value(curr_entry)])


def get_date(entry):
    return entry[0]


def get_value(entry):
    val = entry[1] if len(entry) == 2 else entry[4]
    return val
