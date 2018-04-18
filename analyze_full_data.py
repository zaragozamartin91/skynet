import numpy
import matplotlib.pyplot as plt
import pandas
import math
from skmatrix import normalizer

"dow,dom,month,in_demand,out_demand,prev_holy,pos_holy,minfl"
vars_df = pandas.read_csv('full_data.csv', usecols=[1, 2, 3, 5 ,6 , 7, 8, 9])

