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

import config
from keras import backend as K
import definitions
import os
import pickle
K.set_image_dim_ordering('th')

TRAIN_SAMPLES = 0
TEST_SAMPLES = 0
BATCH_COUNT = 0

def load_globals():
    global TRAIN_SAMPLES
    global TEST_SAMPLES
    global BATCH_COUNT

    TRAIN_SAMPLES, TEST_SAMPLES, BATCH_COUNT \
        = pickle.load(open(os.path.join(definitions.BIN_DATA_DIR, '_metadata.bin'), 'rb'))

def images_per_batch(batch_index):
    if batch_index == BATCH_COUNT:
        return


def load_batch(fpath):
    labels = []
    data = []

    images = pickle.load(open(fpath, 'rb'))

    for img in images:
        labels.append(img[0])
        data.append(img[1])

    return data, labels

def load_data():
    path = definitions.BIN_DATA_DIR

    X_train = np.zeros((TRAIN_SAMPLES, 3, config.IMAGE_SIZE, config.IMAGE_SIZE), dtype="uint8")
    y_train = np.zeros((TRAIN_SAMPLES,), dtype="uint8")

    total_size = 0
    for i in range(1, BATCH_COUNT):
        fpath = os.path.join(path, 'data_batch_%d.bin' % i)
        data, labels = load_batch(fpath)
        batch_size = len(labels)
        X_train[total_size: total_size + batch_size, :, :, :] = data
        y_train[total_size: total_size + batch_size] = labels
        total_size += batch_size

    fpath = os.path.join(path, 'test_batch.bin')
    X_test, y_test = load_batch(fpath)

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    if K.image_dim_ordering() == 'tf':
        X_train = X_train.transpose(0, 2, 3, 1)
        X_test = X_test.transpose(0, 2, 3, 1)

    return (X_train, y_train), (np.array(X_test), np.array(y_test))

def train():
    # fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)

    # load data
    (X_train, y_train), (X_test, y_test) = load_data()

    # normalize inputs from 0-255 to 0.0-1.0
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # one hot encode outputs
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    num_classes = y_test.shape[1]

    # Create the model
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, input_shape=(3, 32, 32), border_mode='same', activation='relu',
                            W_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same', W_constraint=maxnorm(3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu', W_constraint=maxnorm(3)))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    epochs = 25
    lrate = 0.01
    decay = lrate / epochs
    sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    print(model.summary())

    # Fit the model
    model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=epochs, batch_size=32)
    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))

    print("Saving model ...")
    model.save(os.path.join(definitions.MODEL_DATA_DIR, "keras_model.txt"))


if __name__ == '__main__':
    load_globals()
    train()