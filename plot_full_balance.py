import matplotlib.pyplot as plt
import pandas

#   0    1    2    3     4
# INDEX,dom,month,year,balance
balance_df = pandas.read_csv('full_balance.csv', usecols=[1, 2, 3, 4])
balance_ds = balance_df.values

BAL_DOM, BAL_MONTH, BAL_YEAR, BAL_VALUE = range(balance_ds.shape[1])

plt.plot(balance_ds[:, BAL_VALUE], 'r')
# axes = plt.gca()
# axes.set_ylim([0, 1]) # seteo limite en el eje y entre 0 y 1
plt.show()
