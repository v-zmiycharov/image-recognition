import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils

from src.main.common import get_folder

import config
from keras import backend as K
import definitions
import os
import pickle
K.set_image_dim_ordering('th')

def load_globals(folder):
    global TRAIN_BATCHES_COUNT
    global TRAIN_IMAGES_COUNT
    global TEST_BATCHES_COUNT
    global TEST_IMAGES_COUNT

    TRAIN_BATCHES_COUNT, TRAIN_IMAGES_COUNT, TEST_BATCHES_COUNT, TEST_IMAGES_COUNT \
        = pickle.load(open(os.path.join(folder, config.METADATA_FILENAME), 'rb'))


def load_batch(fpath):
    labels = []
    data = []

    images = pickle.load(open(fpath, 'rb'))

    for img in images:
        labels.append(img[0])
        data.append(img[1])

    return data, labels

def get_data(folder, is_train):
    images_count = TRAIN_IMAGES_COUNT if is_train else TEST_IMAGES_COUNT
    batches_count = TRAIN_BATCHES_COUNT if is_train else TEST_BATCHES_COUNT
    prefix = config.TRAIN_BATCH_PREFIX if is_train else config.TEST_BATCH_PREFIX

    X_data = np.zeros((images_count, config.IMAGE_DEPTH, config.IMAGE_SIZE, config.IMAGE_SIZE), dtype="uint8")
    y_data = np.zeros((images_count,), dtype="uint8")

    total_size = 0
    for i in range(batches_count):
        fpath = os.path.join(folder, '{0}{1}.bin'.format(prefix, str(i+1)))
        data, labels = load_batch(fpath)
        batch_size = len(labels)
        X_data[total_size: total_size + batch_size, :, :, :] = data
        y_data[total_size: total_size + batch_size] = labels
        total_size += batch_size

    y_data = np.reshape(y_data, (len(y_data), 1))

    return X_data, y_data

def load_data(folder):
    X_train, y_train = get_data(folder, True)
    X_test, y_test = get_data(folder, False)

    if K.image_dim_ordering() == 'tf':
        X_train = X_train.transpose(0, 2, 3, 1)
        X_test = X_test.transpose(0, 2, 3, 1)

    X_test = np.array(X_test)
    y_test = np.array(y_test)

    # normalize inputs from 0-255 to 0.0-1.0
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    return (X_train, y_train), (X_test, y_test)

def create_simple_model(num_classes):
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, input_shape=(3, config.IMAGE_SIZE, config.IMAGE_SIZE), border_mode='same',
                            activation='relu',
                            W_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same', W_constraint=maxnorm(3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu', W_constraint=maxnorm(3)))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model

def create_complex_model(num_classes):
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, input_shape=(3, config.IMAGE_SIZE, config.IMAGE_SIZE), activation='relu', border_mode='same'))
    model.add(Dropout(0.2))
    model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))
    model.add(Dropout(0.2))
    model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))
    model.add(Dropout(0.2))
    model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(1024, activation='relu', W_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu', W_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))
    return model

def save_model(model, parent_folder):
    json_file = os.path.join(parent_folder, config.MODEL_JSON_FILENAME)
    weights_file = os.path.join(parent_folder, config.MODEL_WEIGHTS_FILENAME)

    model_json = model.to_json()
    with open(json_file, "w") as json_file:
        json_file.write(model_json)
    model.save_weights(weights_file)

def train(folder):
    # fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)

    # load data
    (X_train, y_train), (X_test, y_test) = load_data(folder)

    # one hot encode outputs
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    num_classes = y_test.shape[1]

    # Create the model
    model = create_complex_model(num_classes)
    # Compile model
    decay = config.LEARN_RATE / config.EPOCHS
    sgd = SGD(lr=config.LEARN_RATE, momentum=config.MOMENTUM, decay=decay, nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    print(model.summary())

    # Fit the model
    model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=config.EPOCHS, batch_size=config.BATCH_SIZE, verbose=2)
    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))

    print("Saving model ...")
    save_model(model, folder)


if __name__ == '__main__':
    user_folder = get_folder(definitions.MODELS_DIR)

    for path in os.listdir(user_folder):
        path = os.path.join(user_folder, path)
        if os.path.isdir(path):
            load_globals(path)
            train(path)