import numpy
import pandas
import matplotlib.pyplot as plt

numpy.random.seed(7)

suc = '761'

in_file = 'in_' + suc + '.csv'
out_file = 'out_' + suc + '.csv'
net_file = 'net_' + suc + '.csv'
whole_file = 'whole_demand.csv'

in_df = pandas.read_csv(in_file, usecols=[1])
out_df = pandas.read_csv(out_file, usecols=[1])
net_df = pandas.read_csv(net_file, usecols=[1])
whole_df = pandas.read_csv(whole_file, usecols=[1,2])

in_ds = in_df.values
out_ds = out_df.values
net_ds = net_df.values
whole_ds = whole_df.values

plt.plot(whole_ds[:,0] , 'b-o' ,  label="Demanda entrada verdadera")
plt.plot(in_ds , 'r-o' , label="Demanda entrada predecida")
plt.legend()
plt.show()

plt.plot(whole_ds[:,1] , 'b-o' , label="Demanda salida verdadera")
plt.plot(out_ds , 'r-o' , label="Demanda salida predecida")
plt.legend()
plt.show()

true_diff = whole_ds[:,0] - whole_ds[:,1]
plt.plot(true_diff , 'b-o' , label="Demanda neta verdadera")
plt.plot(net_ds , 'r-o' , label="Demanda neta predecida")
plt.legend()
plt.show()

calc_diff = in_ds - out_ds

plt.plot(true_diff , 'b-o' ,  label="Demanda neta verdadera")
plt.plot(calc_diff, 'r-o' , label="Demanda neta CALCULADA")
plt.plot(net_ds, 'g-o' , label="Demanda neta PREDECIDA")
plt.legend()
plt.show()

