import numpy as np


def remove_max(ds, count):
    if count is 0: return ds
    maxv = ds.argmax()
    return remove_max(np.delete(ds, maxv, axis=0), count - 1)


def remove_min(ds, count):
    if count is 0: return ds
    minv = ds.argmin()
    return remove_min(np.delete(ds, minv, axis=0), count - 1)


def remove_max_on_region(ds, count, start_idx, end_idx):
    a = remove_max(ds[start_idx:end_idx], count)
    return np.vstack([ds[0:start_idx], a, ds[end_idx:]])


def remove_min_on_region(ds, count, start_idx, end_idx):
    a = remove_min(ds[start_idx:end_idx], count)
    return np.vstack([ds[0:start_idx], a, ds[end_idx:]])
