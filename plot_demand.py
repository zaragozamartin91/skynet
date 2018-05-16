import sys
import pandas
import matplotlib.pyplot as plt
import numpy as np

if(len(sys.argv) < 2):
  print('Argumentos: archivo_demanda')
  sys.exit(1)

input_file = sys.argv[1]

#  0    1   2    3     4      5         6          7        8
#INDEX,dow,dom,month,year,in_demand,out_demand,prev_holi,pos_holi
dataframe = pandas.read_csv(input_file, delimiter=',' ,usecols=[5,6], engine='python')
dataset = dataframe.values.astype('float32')

DIN_IDX = 0
DOUT_IDX= 1

din = dataset[:,DIN_IDX] / 1000
dout = dataset[:,DOUT_IDX] / 1000

# plt.plot(np.arange(din.size) , din , label='Entradas')
plt.plot(np.arange(dout.size) , dout , label='salidas')
plt.xlabel('Tiempo desde Jul 1 2014')
plt.ylabel('Dinero en miles')
plt.show()
