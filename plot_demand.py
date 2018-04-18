import sys
import pandas
import matplotlib.pyplot as plt
import numpy as np

if(len(sys.argv) < 2):
  print('Argumentos: archivo_demanda')
  sys.exit(1)

input_file = sys.argv[1]

# we can exclude with the skipfooter argument to pandas.read_csv() set to 3 for the 3 footer lines
dataframe = pandas.read_csv(input_file, delimiter=';' ,usecols=[5,6], engine='python')
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
