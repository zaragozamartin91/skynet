import numpy

def build_win_matrix(dataset, win_size):
    """ 
    Crea una matriz tridimensional en la que cada entrada contiene los datos de ese dia y los datos de los win_size dias anteriores  
    """
    win_matrix = []
    row_count = len(dataset)
    for idx in range(win_size, row_count):
        lower_limit = idx - win_size
        entries = dataset[lower_limit:idx + 1]
        row = []
        for entry in entries:
            row.append(entry)
        win_matrix.append(row)
    return numpy.array(win_matrix)

