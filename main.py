import numpy as np
import matplotlib.pyplot as plt
import pandas
import math

DAY_OF_WEEK=0
DAY_OF_MONTH=1

def dow(days_dataset):  
  return days_dataset[DAY_OF_WEEK]

def dom(days_dataset):
  return days_dataset[DAY_OF_MONTH]


delim = ','

days_dataframe = pandas.read_csv('days.csv', engine='python' , delimiter=delim)
days_dataset = days_dataframe.values
print(days_dataset)

demand_dataframe = pandas.read_csv('demand.csv', engine='python' , delimiter=delim)
demand_dataset = demand_dataframe.values
print(demand_dataset)

print( np.hstack( (days_dataset , demand_dataset) ) )

# datasetX = days_dataset[:,1]
# datasetY = demand_dataset
# plt.plot( datasetX , datasetY )
# plt.xlabel('Dia')
# plt.ylabel('Demanda')
# plt.show()

indexes = [ 0,  1,  2,  3,  4,  5, 8,  9, 10, 11, 12, 13, 14, 15, 16, 19]
days_dataset_h = days_dataset[indexes]
demand_dataset_h = demand_dataset[indexes]

print('hiatus data')
print(days_dataset_h)
print(demand_dataset_h)


