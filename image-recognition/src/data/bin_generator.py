import os
from random import shuffle

import numpy as np
from PIL import Image

import src.config as config
import definitions
from src.data.imagenet_metadata import IMAGES

import pickle
from time import localtime, strftime

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def save_np_to_file(chunk, batch_number, is_train, folder):
    file_name = (config.TRAIN_BATCH_PREFIX if is_train else config.TEST_BATCH_PREFIX) + '{0}.bin'.format(batch_number)

    file_path = os.path.join(folder, file_name)
    pickle.dump(chunk, open(file_path, 'wb'))


def img_to_np(img_path, image_size):
    pic = Image.open(img_path)
    pic = pic.resize((image_size,image_size), Image.ANTIALIAS)

    pic = (np.array(pic))

    r = pic[:, :, 0].flatten()
    g = pic[:, :, 1].flatten()
    b = pic[:, :, 2].flatten()

    return np.array([
            np.reshape(tuple(r), (image_size, image_size)),
            np.reshape(tuple(g), (image_size, image_size)),
            np.reshape(tuple(b), (image_size, image_size))
        ])


def img_to_list(img_path, image_size, label):
    return (
        np.array([label]),
        img_to_np(img_path, image_size)
    )


def load_cross_validation():
    allowed_labels = IMAGES[:config.NUM_CLASSES]

    result_list=[]

    for label in allowed_labels:
        full_dir_path = os.path.join(definitions.IMAGE_NET_DIR, label[1])
        files = os.listdir(full_dir_path)

        for filename in files:
            result_list.append((label[0], os.path.join(full_dir_path, filename)))

    shuffle(result_list)

    all_chunks = list(chunks(result_list, (len(result_list)//config.N_FOLD_CROSS_VALIDATION) + 1))

    stop_range = len(all_chunks) if config.CROSS_VALIDATION_ENABLED else 1
    for i in range(stop_range):
        yield list(inner for index, chunk in enumerate(all_chunks) if index != i for inner in chunk), all_chunks[i]


def save_bins(paths, folder, is_train):
    batch_number = 0

    for chunk in chunks(paths, config.IMAGES_IN_BATCH):
        img_info_chunk = []
        for label_id, path in chunk:
            try:
                img_info_chunk.append(img_to_list(path, config.IMAGE_SIZE, label_id))
            except IndexError:
                print('Index Error: ', path)

        batch_number += 1
        print("Generate {0} batch #{1}".format("train" if is_train else "test", batch_number))
        save_np_to_file(img_info_chunk, batch_number, is_train, folder)

    return batch_number, len(paths)


def handle_paths(train_paths, test_paths, iter_number, folder):
    folder = os.path.join(folder, str(iter_number))
    os.mkdir(folder)

    train_batches_count, train_images_count = save_bins(train_paths, folder, True)
    test_batches_count, test_images_count = save_bins(test_paths, folder, False)

    pickle.dump(
        (train_batches_count, train_images_count, test_batches_count, test_images_count)
        , open(os.path.join(folder, config.METADATA_FILENAME), 'wb'))

def main():
    if not os.path.exists(definitions.MODELS_DIR):
        os.makedirs(definitions.MODELS_DIR)

    current_folder = os.path.join(definitions.MODELS_DIR, strftime("%Y%m%d_%H%M%S", localtime()))
    os.mkdir(current_folder)

    for i, (train_paths, test_paths) in enumerate(load_cross_validation()):
        print('--------- ITERATION {0} ---------'.format(str(i+1)))
        handle_paths(train_paths, test_paths, i+1, current_folder)


if __name__ == '__main__':
    main()




