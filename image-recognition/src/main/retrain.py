import os
import src.config as config
import definitions
import keras.models as k_models
from keras.optimizers import SGD
from src.main.train import load_data
from src.main.train import load_globals
from src.main.common import get_folder
from keras.layers import Dropout
from keras.utils import np_utils

def modify_model(model):
    model.layers[0].trainable = False

    layer = Dropout(0.2)
    model.layers.insert(3, layer)
    layer.build(model.layers[2].output_shape)

    model._flattened_layers = None

def load_model(parent_folder):
    json_file = os.path.join(parent_folder, config.MODEL_JSON_FILENAME)
    weights_file = os.path.join(parent_folder, config.MODEL_WEIGHTS_FILENAME)

    global MODEL

    with open(json_file, 'r') as fr:
        loaded_model_json = fr.read()
    MODEL = k_models.model_from_json(loaded_model_json)
    # load weights into new model
    MODEL.load_weights(weights_file)

    modify_model(MODEL)

    # compile model
    decay = config.LEARN_RATE / config.EPOCHS
    sgd = SGD(lr=config.LEARN_RATE, momentum=config.MOMENTUM, decay=decay, nesterov=False)
    MODEL.compile(loss='sparse_categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

if __name__ == '__main__':
    user_folder = get_folder(definitions.MODELS_DIR)

    for dir_name in os.listdir(user_folder):
        path = os.path.join(user_folder, dir_name)
        if os.path.isdir(path):
            load_globals(path)
            load_model(path)

            (X_train, y_train), (X_test, y_test) = load_data(path)
            y_train = y_train.reshape((-1, 1))
            y_test = y_test.reshape((-1, 1))

            print(MODEL.summary())
            # Fit the model
            MODEL.fit(X_train, y_train, validation_data=(X_test, y_test.reshape((-1, 1))), nb_epoch=config.EPOCHS, batch_size=config.BATCH_SIZE,
                      verbose=2)
            # Final evaluation of the model
            scores = MODEL.evaluate(X_test, y_test, verbose=0)
            print("Accuracy: %.2f%%" % (scores[1] * 100))


