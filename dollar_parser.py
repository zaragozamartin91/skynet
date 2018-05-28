import pandas as pd
import numpy as np
from skmatrix import dollar_matrix


COL_COUNT = 2
dollar_stats_df = pd.read_csv('dollar_stats.csv' , sep=';',usecols=[0,1])
dollar_stats_ds = dollar_stats_df.values

DATE_COL , VALUE_COL = range(COL_COUNT)

# invierto el arreglo
dollar_stats_ds = dollar_stats_ds[::-1]
# __dollar_stats_ds = np.append(__dollar_stats_ds , [dollar_stats_ds[0]], axis=0)
# dollar_stats_ds = __dollar_stats_ds

out_file_path = 'dollar_stats_ord.csv'

cols = ['date','value']
full_df = pd.DataFrame(data=dollar_stats_ds, columns=cols)
full_df.to_csv(out_file_path)


dollar_stats_ds = dollar_matrix.fill_dataset(dollar_stats_ds)
cols = ['date','day','month','year','value']
full_df = pd.DataFrame(data=dollar_stats_ds, columns=cols)
full_df.to_csv(out_file_path)

