from skdate import date_parser
from datetime import datetime

strdate = '2012-10-09T19:00:55Z'
print('Parseando fecha ' + strdate)
v = date_parser.parse_date(strdate)

print(v)

d = date_parser.parse_date_wformat('Jul 1 2014 12:00AM', '%b %d %Y %I:%M%p')
print(d)

d = date_parser.parse_date_wformat('03/21/1991', '%m/%d/%Y')
print(d)

start_date = d['full_date']
end_date = date_parser.add_days(start_date)
print(end_date)

print(date_parser.wrap_date(end_date))