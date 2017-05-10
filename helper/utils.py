import os
import pickle

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
    with open(os.path.join(dir, file_name+ ".p"), "rb") as file:
        return pickle.load(file)
