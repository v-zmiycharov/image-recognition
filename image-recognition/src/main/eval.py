import os
import config
import definitions
import src.data.bin_generator as bin_generator
import keras.models as k_models
import numpy as np
from keras.optimizers import SGD
from data.imagenet_metadata import IMAGES
import pickle
from src.main.train import get_data

def load_model(folder):
    global MODEL
    MODEL = k_models.load_model(os.path.join(folder, config.MODEL_FILENAME))

    decay = config.LEARN_RATE / config.EPOCHS
    sgd = SGD(lr=config.LEARN_RATE, momentum=config.MOMENTUM, decay=decay, nesterov=False)
    MODEL.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

def predict(img_path):
    img_array = bin_generator.img_to_np(img_path, config.IMAGE_SIZE)
    return MODEL.predict(np.array([img_array]))

def predict_label(img):
    predictions = MODEL.predict(img)
    m = max(predictions)
    max_index = [i for i,j in enumerate(predictions) if j == m]
    return max_index

def get_label(index):
    return [item[1] for item in IMAGES if item[0] == index][0]

def format_scores(scores):
    for score_row in scores:
        return "-".join([get_label(index) + ":%.2f" % score for index, score in enumerate(score_row) if score >= 0.01])

def log_acc(header, correct, total):
    print('{0} : {1}% ({2} / {3})'.format(header, 100*(correct/total), correct, total))

def log_info(header, correct_dict, total_dict):
    print('------------ {0} ------------'.format(header))
    correct = 0
    total = 0
    for key in correct_dict:
        correct += correct_dict[key]
        total += total_dict[key]
        log_acc(key, correct_dict[key], total_dict[key])

    log_acc('TOTAL', correct, total)

if __name__ == '__main__':
    total_items = dict()
    correct_items = dict()

    for dir_name in os.listdir(definitions.MODELS_DIR):
        path = os.path.join(definitions.MODELS_DIR, dir_name)
        if os.path.isdir(path):
            load_model(path)
            folder_total_items = dict()
            folder_correct_items = dict()

            X_data, y_data = get_data(path, False)

            for img, label in zip(X_data, y_data):
                if not label in folder_total_items:
                    folder_total_items[label] = 0
                if not label in folder_correct_items:
                    folder_total_items[label] = 0

                folder_total_items[label] += 1

                predicted = predict_label(img)
                if predicted == label:
                    folder_correct_items[label] += 1


            for key, value in folder_total_items.items():
                if key not in total_items:
                    total_items[key] = 0
                total_items[key] += folder_total_items[key]


            for key, value in folder_correct_items.items():
                if key not in correct_items:
                    correct_items[key] = 0
                correct_items[key] += folder_correct_items[key]

            log_info(dir_name, folder_correct_items, folder_total_items)

    log_info('TOTAL', correct_items, total_items)
