from src.data.bin_generator import main as bin_generator_main
from src.data.image_crawler import main as image_crawler_main
from src.main.train import main as train_main
from src.main.eval import main as eval_main

import collections


def init_functions_dict():
    dict = {
        '1': ('Image crawler', image_crawler_main),
        '2': ('Bin generator', bin_generator_main),
        '3': ('Train', train_main),
        '4': ('Eval', eval_main)
    }
    return collections.OrderedDict(sorted(dict.items()))

if __name__ == '__main__':
    functions_dict = init_functions_dict()

    for key, value in functions_dict.items():
        print("{0}: {1}".format(key, value[0]))

    while True:
        key = input('Choose what to do: ')
        if key in functions_dict:
            print('Starting', functions_dict[key][0])
            functions_dict[key][1]()
            exit()