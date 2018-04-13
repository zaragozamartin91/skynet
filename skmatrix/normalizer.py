import numpy as np

def normalize_int(dataset):
  """ Normaliza un dataset de una sola fila que contenga valores enteros """
  ds0 = dataset.astype('int32')
  minv = ds0.min()
  maxv = ds0.max()
  diff = maxv - minv
  ds1 = ds0 - minv
  ds2 = ds1 / diff
  return ds2.astype('float32')

def normalize_float(dataset):
  """ Normaliza un dataset de una sola fila que contenga valores punto flotante """
  ds0 = dataset.astype('float64')
  minv = ds0.min()
  maxv = ds0.max()
  diff = maxv - minv
  ds1 = ds0 - minv
  ds2 = ds1 / diff
  return ds2.astype('float64')

def normalize_any(dataset):
  """ Normaliza un dataset que contenga valores de cualquier tipo """
  ds0 = dataset
  minv = ds0.min()
  maxv = ds0.max()
  diff = maxv - minv
  ds1 = ds0 - minv
  ds2 = ds1 / diff
  return ds2.astype('float32')
