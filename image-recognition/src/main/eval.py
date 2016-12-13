import os
import config
import definitions
import src.data.bin_generator as bin_generator
import keras.models as k_models
import numpy as np
from keras.optimizers import SGD
from data.imagenet_metadata import IMAGES

MODEL = None

def load_model():
    global MODEL
    MODEL = k_models.load_model(os.path.join(definitions.MODEL_DATA_DIR, "keras_model1.txt"))

    decay = config.LEARN_RATE / config.EPOCHS
    sgd = SGD(lr=config.LEARN_RATE, momentum=config.MOMENTUM, decay=decay, nesterov=False)
    MODEL.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

def predict(img_path):
    img_array = bin_generator.img_to_np(img_path, config.IMAGE_SIZE)
    return MODEL.predict(np.array([img_array]))

def get_label(index):
    return [item[1] for item in IMAGES if item[0] == index][0]

def format_scores(scores):
    for score_row in scores:
        return "-".join([get_label(index) + ":%.2f" % score for index, score in enumerate(score_row) if score >= 0.01])

if __name__ == '__main__':
    load_model()

    for tuple in IMAGES[:config.NUM_CLASSES]:
        animal = tuple[1]
        print('----------', animal.upper(), '----------')

        dir_path = os.path.join(definitions.IMAGE_NET_DIR, animal, definitions.TEST_DIR_NAME)

        for file_name in [file_name for file_name in os.listdir(dir_path)]:
            path = os.path.join(dir_path, file_name)
            try:
                scores = predict(path)
                print(format_scores(scores), file_name)
            except IndexError as index_err:
                print('---------- ERROR: ', file_name, '----------')

        print('\n\n\n\n\n')