import os
import src.config as config
import definitions
import src.data.bin_generator as bin_generator
import keras.models as k_models
import numpy as np
from keras.optimizers import SGD
from data.imagenet_metadata import IMAGES
from src.main.train import load_data
from src.main.train import load_globals
from src.main.common import get_folder

def load_model(parent_folder):
    json_file = os.path.join(parent_folder, config.MODEL_JSON_FILENAME)
    weights_file = os.path.join(parent_folder, config.MODEL_WEIGHTS_FILENAME)

    global MODEL

    with open(json_file, 'r') as fr:
        loaded_model_json = fr.read()
    MODEL = k_models.model_from_json(loaded_model_json)
    # load weights into new model
    MODEL.load_weights(weights_file)

    # compile model
    decay = config.LEARN_RATE / config.EPOCHS
    sgd = SGD(lr=config.LEARN_RATE, momentum=config.MOMENTUM, decay=decay, nesterov=False)
    MODEL.compile(loss='sparse_categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

def predict(img_path):
    img_array = bin_generator.img_to_np(img_path, config.IMAGE_SIZE)
    return MODEL.predict(np.array([img_array]))

def predict_label(img):
    predictions = MODEL.predict(np.array([img]))
    predictions = predictions[0]
    m = max(predictions)
    max_index = [i for i,j in enumerate(predictions) if j == m]
    return max_index[0]

def get_label(index):
    return [item[1] for item in IMAGES if item[0] == index][0]

def format_scores(scores):
    for score_row in scores:
        return "-".join([get_label(index) + ":%.2f" % score for index, score in enumerate(score_row) if score >= 0.01])

def log_acc(header, correct, total):
    print('{0}: {1}%     {2}/{3}'.format(header, "%.2f" % (100*(correct/total)), correct, total))

def log_info(header, correct_dict, total_dict):
    if config.CROSS_VALIDATION_ENABLED:
        print('------------ {0} ------------'.format(header))
    correct = 0
    total = 0
    for label in correct_dict:
        correct += correct_dict[label]
        total += total_dict[label]
        log_acc(get_label(label), correct_dict[label], total_dict[label])

    log_acc('TOTAL', correct, total)
    print()

def init_dict():
    result = dict()
    for tuple in IMAGES[:config.NUM_CLASSES]:
        result[tuple[0]] = 0
    return result

def main():
    user_folder = get_folder(definitions.MODELS_DIR)

    total_items = init_dict()
    correct_items = init_dict()

    for dir_name in os.listdir(user_folder):
        path = os.path.join(user_folder, dir_name)
        if os.path.isdir(path):
            load_globals(path)
            load_model(path)

            folder_total_items = init_dict()
            folder_correct_items = init_dict()

            (_, _), (X_data, y_data) = load_data(path)

            '''
            scores = MODEL.evaluate(X_data, y_data.reshape((-1,1)), verbose=0)
            print("Accuracy: %.2f%%" % (scores[1] * 100))
            '''

            for img, label_np in zip(X_data, y_data):
                label = label_np[0]

                folder_total_items[label] += 1

                predicted = predict_label(img)
                if predicted == label:
                    folder_correct_items[label] += 1


            for key, value in folder_total_items.items():
                total_items[key] += folder_total_items[key]

            for key, value in folder_correct_items.items():
                correct_items[key] += folder_correct_items[key]

            log_info(dir_name, folder_correct_items, folder_total_items)

    if config.CROSS_VALIDATION_ENABLED:
        log_info('TOTAL', correct_items, total_items)


if __name__ == '__main__':
    main()
