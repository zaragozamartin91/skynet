import pandas as pd
import numpy as np

COL_COUNT = 2
dollar_stats_df = pd.read_csv('dollar_stats.csv' , sep=';',usecols=[0,1])
dollar_stats_ds = dollar_stats_df.values

DATE_COL , VALUE_COL = range(COL_COUNT)

# invierto el arreglo
dollar_stats_ds = dollar_stats_ds[::-1]

