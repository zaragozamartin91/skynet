import numpy


def categorize_real_w_equal_frames(dataset, cat_count=1, cat_col=0):
    """ Categoriza una columna de dataset creando categorias de franjas de mismo tamano.
    El tamano de las franjas de categoria es igual para todas, pero la cantidad de elementos que van a cada franja es variable
    Parametros
    ----------
    dataset: ds a categorizar
    cat_count: cantidad de categorias a crear
    cat_col: columna del ds a categorizar

    Retorno
    -------
    categorized: dataset con columna indicada categorizada
    cat_bags: lista de listas que contiene categoria de los elementos """
    sub_ds = dataset[:, cat_col]  # obtengo los valores a categorizar
    min_val = sub_ds.min()
    diff = sub_ds.max() - min_val
    cat_frame_size = diff / cat_count

    cat_bags = numpy.zeros(cat_count)
    for idx in range(cat_count):
        cat_upper_bound = min_val + cat_frame_size * (idx + 1)
        cat_bags[idx] = cat_upper_bound

    categorized = dataset.copy()
    for idx in range(len(dataset)):
        item = dataset[idx, cat_col]
        item_category = categorize_value(item, cat_frame_size, min_val)
        categorized[idx, cat_col] = item_category

    return categorized, cat_bags.reshape([cat_count,1])


def categorize_int(dataset, cat_col=0):
    categorized = dataset.copy()
    a = categorized[:, cat_col] - categorized[:, cat_col].min()
    categorized[:, cat_col] = a
    return categorized


def one_hot(categorized, cat_col=0):
    """ Divide un dataset de categorias en un tensor de valores 0 y 1 """
    sub_ds = categorized[:, cat_col]  # obtengo los valores a dividir
    sub_ds = sub_ds - sub_ds.min()
    cat_count = int(sub_ds.max() + 1)
    ds_size = len(categorized)
    arr = numpy.zeros((ds_size, cat_count))
    for idx in range(ds_size):
        cat = int(categorized[idx, cat_col])
        arr[idx, cat] = 1
    return arr


def categorize_value(value, cat_frame_size, minv=0, curr=0):
    v = minv + cat_frame_size * (curr + 1)
    if value <= v: return curr
    else: return categorize_value(value, cat_frame_size, minv, curr + 1)


def categorize_arr(arr, cat_frame_size, minv=0):
    a = numpy.array(arr).flatten()
    for idx in range(len(a)):
        item_value = a[idx]
        item_category = categorize_value(item_value, cat_frame_size, minv)
        a[idx] = item_category
    return a

