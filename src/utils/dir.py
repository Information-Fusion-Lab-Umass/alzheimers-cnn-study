# Utility functions related to directories

import os

def mkdir(path):
    '''
    Creates the specified directory if it is valid and does not exist.

    Note: the location of the directory will dependent on where this is invoked. For example, calling `mkdir("outputs")` in `disease_forecasting/src/cae/` will create `disease_forecasting/src/cae/outputs/`.

    Args:
        path (string): The directory or folder to create.
    '''
    if not os.path.exists(path):
        os.mkdir(path)

def lsdir(path):
    '''
    Lists all of the files in the specified path, excluding files that start with a ".".

    Args:
        path (string):
    '''
    files = os.listdir(path)
    return list(filter(lambda name: name[0] != ".", files))
