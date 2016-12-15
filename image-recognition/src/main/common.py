import os
from pprint import pprint

def init_folders_dict(parent_folder):
    result = dict()

    index = 1
    for folder in os.listdir(parent_folder):
        path = os.path.join(parent_folder, folder)
        if os.path.isdir(path):
            result[str(index)] = folder
            index += 1

    return result


def get_folder(parent_folder):
    folders_dict = init_folders_dict(parent_folder)
    pprint(folders_dict, width=1)
    while True:
        index = input('Index of folder: ')
        if index in folders_dict:
            return os.path.join(parent_folder, folders_dict[index])
