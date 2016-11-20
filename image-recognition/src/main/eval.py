import os
import config
import definitions
import src.data.bin_generator as bin_generator
import keras.models as k_models
import numpy as np

MODEL = None

def load_model():
    global MODEL
    MODEL = k_models.load_model(os.path.join(definitions.MODEL_DATA_DIR, "keras_model.txt"))

def predict(img_path):
    img_array = bin_generator.img_to_np(img_path, config.IMAGE_SIZE)
    return MODEL.predict(np.array([img_array]))

if __name__ == '__main__':
    load_model()

    img_paths = [
        os.path.join(definitions.IMAGE_NET_DIR, "bear", definitions.TEST_DIR_NAME, "n02131653_577.JPEG"),
        os.path.join(definitions.IMAGE_NET_DIR, "antelope", definitions.TEST_DIR_NAME, "n02419796_154.JPEG"),
        os.path.join(definitions.IMAGE_NET_DIR, "antelope", definitions.TEST_DIR_NAME, "n02419796_1311.JPEG"),
        os.path.join(definitions.IMAGE_NET_DIR, "bear", definitions.TEST_DIR_NAME, "n02131653_650.JPEG")
    ]

    for path in img_paths:
        print(predict(path), ": ", path)