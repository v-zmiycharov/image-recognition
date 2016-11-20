import os
from random import shuffle

import numpy as np
from PIL import Image

import config
import definitions
from data.imagenet_metadata import IMAGES

import pickle

def clear_dir(dir_path):
    for file in os.listdir(dir_path):
        path = os.path.join(dir_path, file)
        if os.path.isfile(path):
            os.unlink(path)

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def save_np_to_file(chunk, batch_number = 0, is_train = True):
    file_name = 'data_batch_{0}.bin'.format(batch_number)

    if not is_train:
        file_name = 'test_batch.bin'

    file_path = os.path.join(definitions.BIN_DATA_DIR, file_name)
    pickle.dump(chunk, open(file_path, 'wb'))

def img_to_list(img_path, image_size, label):
    pic = Image.open(img_path)
    pic = pic.resize((image_size,image_size), Image.ANTIALIAS)

    pic = (np.array(pic))

    r = pic[:, :, 0].flatten()
    g = pic[:, :, 1].flatten()
    b = pic[:, :, 2].flatten()

    return (
        np.array([label]),
        np.array([
            np.reshape(tuple(r), (image_size, image_size)),
            np.reshape(tuple(g), (image_size, image_size)),
            np.reshape(tuple(b), (image_size, image_size))
        ])
    )


def load_dataset(num_classes, image_size, images_count = 0, is_train = True):
    dir_name = definitions.IMAGES_DIR_NAME if is_train else definitions.TEST_DIR_NAME

    allowed_labels = IMAGES[:num_classes]

    result_list=[]

    for label in allowed_labels:
        full_dir_path = os.path.join(definitions.IMAGE_NET_DIR, label[1], dir_name)
        files = os.listdir(full_dir_path)
        if images_count > 0:
            files = files[:images_count]

        for filename in files:
            try:
                result_list.append(img_to_list(os.path.join(full_dir_path, filename), image_size, label[0]))
            except IndexError as err:
                print(os.path.join(full_dir_path, filename))

    if is_train:
        shuffle(result_list)

    batches_count = num_classes if is_train else 1
    items_in_batch = len(result_list) // batches_count

    batch_number = 0

    for chunk in chunks(result_list, items_in_batch):
        batch_number += 1
        print("Generate {0} batch #{1}".format("train" if is_train else "test", batch_number))
        save_np_to_file(chunk, batch_number, is_train)

    if is_train:
        print(str(batch_number) + " batches")
        print(str(len(result_list)) + " train items")
    else:
        print(str(len(result_list)) + " test items")

    return batch_number, len(result_list)

if __name__ == '__main__':
    clear_dir(definitions.BIN_DATA_DIR)
    batch_count, train_items = load_dataset(config.NUM_CLASSES, config.IMAGE_SIZE)
    temp, test_items = load_dataset(config.NUM_CLASSES, config.IMAGE_SIZE, is_train = False)

    pickle.dump(
        (train_items, test_items, batch_count)
        , open(os.path.join(definitions.BIN_DATA_DIR, '_metadata.bin'), 'wb'))



