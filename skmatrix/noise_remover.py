import numpy as np


def remove_max(ds, count):
    if count is 0: return ds
    maxv = ds.argmax()
    return remove_max(np.delete(ds, maxv), count - 1)


def remove_min(ds, count):
    if count is 0: return ds
    minv = ds.argmin()
    return remove_min(np.delete(ds, minv), count - 1)

