import os
from random import shuffle

import numpy as np
from PIL import Image

import definitions
from data.imagenet_metadata import IMAGES


def img_to_numpy(img_path, image_size, is_reshape = False):
    pic = Image.open(img_path)
    pic = pic.resize((image_size,image_size), Image.ANTIALIAS)
    np_array = np.array(pic)

    if is_reshape:
        np_array = np_array.reshape((3, image_size, image_size))

    return np_array

def shuffle_dependent_lists(list1, list2):
    list1_shuf = []
    list2_shuf = []
    index_shuf = list(range(len(list1)))
    shuffle(index_shuf)
    for i in index_shuf:
        list1_shuf.append(list1[i])
        list2_shuf.append(list2[i])

    return (list1_shuf, list2_shuf)


def load_dataset(num_classes, image_size, images_count = 0, is_train = True, is_reshape = False):
    dir_name = definitions.IMAGES_DIR_NAME if is_train else definitions.TEST_DIR_NAME
    X_data = []
    y_data = []

    allowed_labels = IMAGES[:num_classes]

    for label in allowed_labels:
        full_dir_path = os.path.join(definitions.IMAGE_NET_DIR, label[1], dir_name)
        files = os.listdir(full_dir_path)
        if images_count > 0:
            files = files[:images_count]

        for filename in files:
            try:
                image = img_to_numpy(os.path.join(full_dir_path, filename), image_size, is_reshape)
            except ValueError as ve:
                print("----- VALUE ERROR -----", os.path.join(full_dir_path, filename))
                continue

            X_data.append(image)
            y_data.append(label[0])

    if is_train:
        (X_data, y_data) = shuffle_dependent_lists(X_data, y_data)

    return np.array(X_data), np.array(y_data)

