import numpy as np

def load_small_data():
    object = np.load("small_data.npz")
    return object

def load_full_data():
    object = np.load('mnistTrain.npz')
    return object

def save_data(name, *data, **attributes): # name is a string, which will be the file name
    """
    name is the file name
    data is a number of data items
    attributes is a keyword list with named data, i.e. name=data_item
    """
    np.savez(name, *data, **attributes)

