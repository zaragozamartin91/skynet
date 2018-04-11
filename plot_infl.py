import pandas
import matplotlib.pyplot as plt
import numpy as np

# we can exclude with the skipfooter argument to pandas.read_csv() set to 3 for the 3 footer lines
dataframe = pandas.read_csv('inflation_s2010.csv', usecols=[0,2,3], engine='python')
dataset = dataframe.values

MONTH = 1
YEAR= 2

month_infl = dataset[:,MONTH]
year_infl = dataset[:,YEAR]

plt.plot(np.arange(month_infl.size) , month_infl , label='Mensual')
plt.plot(np.arange(year_infl.size) , year_infl , label='Anual')
plt.xlabel('Tiempo desde 2010')
plt.ylabel('Inflacion')
plt.show()
