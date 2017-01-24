from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import ZeroPadding2D
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm

def create_vgg16_model(num_classes, image_size):
    fs = 3
    fc = 64
    
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(3, image_size, image_size)))
    model.add(Convolution2D(fc, fs, fs, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(fc, fs, fs, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(fc*2, fs, fs, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(fc*2, fs, fs, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(fc*4, fs, fs, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(fc*4, fs, fs, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(fc*4, fs, fs, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(fc*8, fs, fs, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(fc*8, fs, fs, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(fc*8, fs, fs, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(fc*8, fs, fs, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(fc*8, fs, fs, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(fc*8, fs, fs, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model

def create_simple_model(num_classes, image_size):
    fs = 3
    fc = 32

    model = Sequential()
    model.add(Convolution2D(fc, fs, fs, input_shape=(3, image_size, image_size), border_mode='same',
                            activation='relu',
                            W_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Convolution2D(fc, fs, fs, activation='relu', border_mode='same', W_constraint=maxnorm(3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu', W_constraint=maxnorm(3)))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model

def create_complex_model(num_classes, image_size):
    fs = 3
    fc = 32

    model = Sequential()
    model.add(Convolution2D(fc, fs, fs, input_shape=(3, image_size, image_size), activation='relu', border_mode='same'))
    model.add(Dropout(0.2))
    model.add(Convolution2D(fc, fs, fs, activation='relu', border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(fc*2, fs, fs, activation='relu', border_mode='same'))
    model.add(Dropout(0.2))
    model.add(Convolution2D(fc*2, fs, fs, activation='relu', border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(fc*4, fs, fs, activation='relu', border_mode='same'))
    model.add(Dropout(0.2))
    model.add(Convolution2D(fc*4, fs, fs, activation='relu', border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(1024, activation='relu', W_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu', W_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))
    return model


def create_sigmoid_model(num_classes, image_size):
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, input_shape=(3, image_size, image_size), border_mode='same',
                            activation='relu',
                            W_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same', W_constraint=maxnorm(3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu', W_constraint=maxnorm(3)))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='sigmoid'))
    return model
