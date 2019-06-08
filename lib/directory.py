'''Helper methods for directory-related tasks.
'''
import os
import shutil

def mkdir(path):
    '''Creates the specified directory if it is valid and does not exist.

    The location of the directory will dependent on where this is invoked. For example, calling `mkdir("outputs")` in `CorefQA/` will create `CorefQA/outputs`.
    '''
    if not os.path.exists(path):
        os.mkdir(path)

def lsdir(path):
    ''' Lists all of the files in the specified path, excluding files that start with a ".".
    '''
    files = os.listdir(path)
    return list(filter(lambda name: name[0] != ".", files))

def rmfile(path):
    '''Removes a file if it exists.
    '''
    if os.path.isfile(path):
        os.remove(path)

def rmdir(path):
    '''Removes a directory if it exists.
    '''
    if os.path.exists(path):
        shutil.rmtree(path, ignore_errors=True)
