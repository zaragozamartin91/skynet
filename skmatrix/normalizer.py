import numpy as np


def normalize_dataset(dataset):
    """ Normaliza un dataset """
    col_count = dataset.shape[1]
    for col in range(col_count):
        sub_ds = dataset[:, col]
        max_value = float(max(sub_ds))
        min_value = float(min(sub_ds))
        diff = max_value - min_value
        n_ds = sub_ds - min_value
        n_ds = n_ds / diff
        dataset[:, col] = n_ds
    return dataset

def de_normalize_dataset(normalized_ds, original_ds):
    """ 
    Des-normaliza un dataset a partir del dataset original (completo) 
    :param normalized_ds : Dataset normalizado a des-normalizar
    :param original_ds : Dataset ORIGINAL (no particionado) del cual obtener los valores originales
    :return : Dataset des-normalizado
    """
    col_count = original_ds.shape[1]
    for col in range(col_count):
        original_sub_ds = original_ds[:, col]
        min_value = min(original_sub_ds)
        max_value = max(original_sub_ds)
        diff = max_value - min_value
        a = normalized_ds[:, col] * diff
        b = a + min_value
        normalized_ds[:, col] = b
    return normalized_ds


def normalize_dataset_w_normaldist(dataset):
    """ Normaliza un dataset """
    col_count = dataset.shape[1]
    for col in range(col_count):
        sub_ds = dataset[:, col]
        meanv = sub_ds.mean()
        stdv = sub_ds.std()
        n_ds = (sub_ds - meanv) / stdv
        dataset[:, col] = n_ds
    return dataset

def de_normalize_dataset_w_normaldist(dataset, original_ds):
    """ Normaliza un dataset """
    col_count = dataset.shape[1]
    for col in range(col_count):
        sub_ds = original_ds[:, col]
        meanv = sub_ds.mean()
        stdv = sub_ds.std()
        n_ds = dataset * stdv + meanv
        dataset[:, col] = n_ds
    return dataset
