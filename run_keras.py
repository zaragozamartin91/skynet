import numpy
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from skmatrix import normalizer


def normalize_dataset(dataset):
  cols = dataset.shape[1]
  for col in range(cols):
    sub_ds = dataset[:,col]
    n_ds = normalizer.normalize_float(sub_ds)
    dataset[:,col] = n_ds
  return dataset

def append_prev_demand(vars_ds,demand_ds):
  in_demands = [(0,)]
  out_demands = [(0,)]
  row_count = len(vars_ds)
  for i in range(row_count):
    if(i == 0): continue
    prev_entry = demand_ds[i-1]
    in_demands.append( (prev_entry[0],) )
    out_demands.append( (prev_entry[1],) )
  return numpy.hstack( (vars_ds , numpy.array(in_demands) , numpy.array(out_demands)) )
    

numpy.random.seed(7)

# vars_df = pandas.read_csv('full_data.csv', usecols=[1, 2, 3, 7, 8, 9])
vars_df = pandas.read_csv('full_data.csv', usecols=[1, 2, 3, 7, 8])
demand_df = pandas.read_csv('full_data.csv', usecols=[5,6])

vars_ds = vars_df.values.astype('float64')
demand_ds = demand_df.values.astype('float64')

vars_ds = append_prev_demand(vars_ds , demand_ds)

# PRUEBA REMOVIENDO LOS REGISTROS CON DEMANDA 0
# t = demand_ds[:,0]
# b = t > 1.0
# vars_ds = vars_ds[b]
# demand_ds = demand_ds[b]

# Asigno valores de 0 a 1 a todas las entradas
normalize_dataset(vars_ds)
normalize_dataset(demand_ds)

# The code below calculates the index of the split point and separates the data into the training datasets with 67% of the observations that we can use to train our model, leaving the remaining 33% for testing the model
train_size = int(len(vars_ds) * 0.67)
test_size = len(vars_ds) - train_size

# Creo el dataset de entrenamiento. EL mismo toma las filas
# del 0 a train_size (no inclusive) y todas las columnas (rango ':')
train_vars = vars_ds[0:train_size,:]
train_demand = demand_ds[0:train_size,:]

test_vars = vars_ds[train_size:len(vars_ds),:]
test_demand = demand_ds[train_size:len(demand_ds),:]


# Keras de manera implicita trabaja siempre con una capa de entrada
# input_dim determina la cantidad de neuronas de la capa de entrada
# el primer parametro de 'Dense' es la cantidad de neuronas de la capa oculta
# una red densa o Dense es aquella en la que todas las neuronas de una capa N estan conectadas con todas las de la capa N+1
# el modelo a crear es Multilayer Perceptron 
# la cantidad de neuronas de la capa de salida debe coincidir con la cantidad de valores a predecir
# Si la ultima capa tiene una funcion de activacion, entonces estamos modelando un problema de CLASIFICACION / CLUSTERIZACION en vez de uno de PREDICCION
# epochs es el número de pasadas por todo el conjunto de datos de entrenamiento
# batch_size es el número de muestras que se usan para calcular una actualización de los pesos

input_dim = train_vars.shape[1] # la cantidad de neuronas de input es igual a la cantidad de columnas del dataset de entrada
model = Sequential()
model.add(Dense(20, input_dim=input_dim, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(2, activation='relu'))
opt = optimizers.adam(lr=0.001)
# model.compile(loss='binary_crossentropy', optimizer=opt)
model.compile(loss='mean_squared_error', optimizer=opt)
model.fit(train_vars, train_demand, epochs=200, batch_size=10, verbose=2)


# Estimate model performance
trainScore = model.evaluate(train_vars, train_demand, verbose=0)
print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore, math.sqrt(trainScore)))
testScore = model.evaluate(test_vars, test_demand, verbose=0)
print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore, math.sqrt(testScore)))


# trainPredict = model.predict(train_vars)
testPredict = model.predict(test_vars)

# PLOTEO DE LA DEMANDA DE ENTRADA JUNTO CON TODOS LOS DATOS -----------------------------------------
# ind = demand_ds[:,0]
# ind_testPredict = numpy.empty_like(ind)
# ind_testPredict[:] = numpy.nan
# ind_testPredict[train_size:] = testPredict[:,0]

# plt.plot(ind)
# plt.plot(ind_testPredict)


# PLOTEO DE LA DEMANDA DE SALIDA JUNTO CON TODOS LOS DATOS -----------------------------------------
# outd = demand_ds[:,1]
# outd_testPredict = numpy.empty_like(outd)
# outd_testPredict[:] = numpy.nan
# outd_testPredict[train_size:] = testPredict[:,1]

# plt.plot(outd)
# plt.plot(outd_testPredict)


# PLOTEO DE LA DEMANDA DE SALIDA JUNTO CON LA PORCION DE INCUMBENCIA DE LOS DATOS ORIGINALES -----------------------------------------
outd = demand_ds[train_size:,1]
outd_testPredict = numpy.empty_like(outd)
outd_testPredict[:] = testPredict[:,1]
plt.plot(outd)
plt.plot(outd_testPredict)


plt.show()


# trainPredictPlot = numpy.empty_like(dataset.ravel())
# trainPredictPlot[:] = numpy.nan
# trainPredictPlot[look_back:len(trainPredict)+look_back] = trainPredict.ravel()

# print('trainPredictPlot:')
# print(trainPredictPlot)


# testPredictPlot = numpy.empty_like(dataset)
# testPredictPlot[:, :] = numpy.nan
# testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict


# plt.plot(dataset)
# plt.plot(trainPredictPlot)
# plt.plot(testPredictPlot)
# plt.show()

