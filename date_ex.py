from date_parser import parser
from datetime import datetime

strdate = '2012-10-09T19:00:55Z'
print('Parseando fecha ' + strdate)
v = parser.parse_date(strdate)

print(v)

d = parser.parse_date_wformat('Jul 1 2014 12:00AM', '%b %d %Y %I:%M%p')
print(d)

d = parser.parse_date_wformat('03/21/1991', '%m/%d/%Y')
print(d)

start_date = d['full_date']
end_date = parser.add_days(start_date)
print(end_date)

print(parser.wrap_date(end_date))