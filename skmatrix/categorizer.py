import numpy


def categorize_item(value, cat_bags):
    category = len(cat_bags) - 1  # por defecto devuelvo la ultima categoria
    for idx in range(len(cat_bags)):
        cat_bag = cat_bags[idx]
        if value <= cat_bag.max():
            category = idx
            break
    return category


def categorize_real_w_equal_dist(dataset, cat_count, cat_col=0):
    """ Categoriza una columna de dataset creando categorias con igual distribucion de elementos.
    El tamano de cada franja de categoria es variable pero la cantidad de elementos en cada categoria es la misma.
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
    a = numpy.sort(sub_ds)  # obtengo los valores ordenados del ds a categorizar
    cat_size = int(len(a) / cat_count)  # defino el tamano que debe tener cada categoria
    cat_bags = []
    last_cat = cat_count - 1
    for i in range(cat_count):
        start_idx = i * cat_size
        end_idx = start_idx + cat_size if i < last_cat else len(a)
        cat_bags.append(a[start_idx:end_idx])
    categorized = dataset.copy()
    for idx in range(len(categorized)):
        item = categorized[idx, cat_col]
        item_category = categorize_item(item, cat_bags)
        categorized[idx, cat_col] = item_category
    return categorized, cat_bags


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
    cat_bags = []
    min_val = sub_ds.min()
    diff = sub_ds.max() - min_val
    cat_gap = diff / cat_count
    for i in range(cat_count):
        cat_val = min_val + cat_gap * (i + 1)
        cat_bags.append(numpy.array([cat_val]))
    categorized = dataset.copy()
    for idx in range(len(categorized)):
        item = categorized[idx, cat_col]
        item_category = categorize_item(item, cat_bags)
        categorized[idx, cat_col] = item_category
    return categorized, cat_bags



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
