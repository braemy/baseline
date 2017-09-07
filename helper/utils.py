import os
import pickle

import datetime
import time

import glob
import yaml

from helper.constants import dataset_type


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


def load_pickle(file_name, encoding="utf-8"):
    with open(file_name, "rb") as file:
        return pickle.load(file, encoding=encoding)


def load_parameters(name):
    with open("parameters.yml", 'r') as ymlfile:
        return yaml.load(ymlfile)[name]

def load_parameters_tf():
    with open("parameters_crf_tf.yml", 'r') as ymlfile:
        return yaml.load(ymlfile)

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

def get_training_file(parameters, language, dataset_t):
    """
    return the training file given the language and type of file
    :param language: language of the file
    :type language: str
    :param type: type of the file (coNLL, wikiNER, wd)
    :type type: constants.dataset_type
    :return: training file
    :rtype: str
    """
    if dataset_t == dataset_type.conll:
        assert language in ['en', 'de']
        if language == "en":
            l = "eng"
        elif language == "de":
            l = "deu"
        else:
            raise("Unrecognized language")
        return os.path.join(parameters['path'][dataset_t.name], "{}.train".format(l))
    if dataset_t == dataset_type.wikiner:
        return os.path.join(parameters['path'][dataset_t.name], "aij-wikiner-{}-wp2-simplified.train".format(language))
    if dataset_t == dataset_type.new_dataset:
        if language in ['fr', 'it', 'en', 'de']:
            return os.path.join(parameters['path'][dataset_t.name], language, "wp3",  "combined_wp3_0.3.train")
        else:
            return os.path.join(parameters['path'][dataset_t.name], language, "wp3",  "combined_wp3_1.0.train")


def keep_only_best_model(model_folder, best_model_epoch, max_number_epoch):
    print("Remove all the model and keep only the best one")
    for i in range(0,max_number_epoch):
        if i != best_model_epoch:
            model_path = os.path.join(model_folder, 'model_{0:05d}.ckpt*'.format(i))
            for file in glob.glob(model_path):
                os.remove(file)

def create_folder_if_not_exists(directory):
    '''
    Create the folder if it doesn't exist already.
    '''
    if not os.path.exists(directory):
        os.makedirs(directory)

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
