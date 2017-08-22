import os
import pickle

import datetime
import time
import yaml



def create_folder(folder_name):
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)


def create_recursive_folder(list_folder_name):
    folder_path = "."
    for folder_name in list_folder_name:
        folder_path = os.path.join(folder_path, folder_name)
        create_folder(folder_path)


def pickle_data(data, dir, file_name):
    with open(os.path.join(dir, file_name + ".p"), "wb") as file:
        pickle.dump(data, file)


def load_pickle(dir, file_name):
    with open(os.path.join(dir, file_name + ".p"), "rb") as file:
        return pickle.load(file)


def load_parameters(name):
    with open("parameters.yml", 'r') as ymlfile:
        return yaml.load(ymlfile)[name]

def get_current_time_in_miliseconds():
    '''
    http://stackoverflow.com/questions/5998245/get-current-time-in-milliseconds-in-python
    '''
    return(get_current_time_in_seconds() + '-' + str(datetime.datetime.now().microsecond))

def get_current_time_in_seconds():
    '''
    http://stackoverflow.com/questions/415511/how-to-get-current-time-in-python
    '''
    return(time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()))


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
