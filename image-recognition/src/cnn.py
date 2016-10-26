import os
import numpy as np
import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from PIL import Image
import definitions
from src.image_net import IMAGES

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

IMAGE_SIZE = 28

def img_to_numpy(img_path):
    pic = Image.open(img_path)
    pic = pic.resize((IMAGE_SIZE,IMAGE_SIZE), Image.ANTIALIAS)
    np_array = np.array(pic)
    return np_array.reshape((3,IMAGE_SIZE,IMAGE_SIZE))

def load_dataset():
    X_train = []
    X_test = []
    y_train = []
    y_test = []

    allowed_labels = IMAGES[:2]

    for label in allowed_labels:
        train_dir = os.path.join(definitions.IMAGE_NET_DIR, label[1], definitions.IMAGES_DIR_NAME)
        test_dir = os.path.join(definitions.IMAGE_NET_DIR, label[1], definitions.TEST_DIR_NAME)
        for filename in os.listdir(train_dir):
            try:
                image = img_to_numpy(os.path.join(train_dir, filename))
            except ValueError as ve:
                print(os.path.join(train_dir, filename))
                continue

            X_train.append(image)
            y_train.append(label[0])
        for filename in os.listdir(test_dir):
            try:
                image = img_to_numpy(os.path.join(test_dir, filename))
            except ValueError as ve:
                print(os.path.join(test_dir, filename))
                continue

            X_test.append(image)
            y_test.append(label[0])

    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)

X_train, y_train, X_test, y_test = load_dataset()

net1 = NeuralNet(
    layers=[('input', layers.InputLayer),
            ('conv2d1', layers.Conv2DLayer),
            ('maxpool1', layers.MaxPool2DLayer),
            ('conv2d2', layers.Conv2DLayer),
            ('maxpool2', layers.MaxPool2DLayer),
            ('dropout1', layers.DropoutLayer),
            ('dense', layers.DenseLayer),
            ('dropout2', layers.DropoutLayer),
            ('output', layers.DenseLayer),
            ],
    # input layer
    input_shape=(None, 3, IMAGE_SIZE, IMAGE_SIZE),
    # layer conv2d1
    conv2d1_num_filters=32,
    conv2d1_filter_size=(5, 5),
    conv2d1_nonlinearity=lasagne.nonlinearities.rectify,
    conv2d1_W=lasagne.init.GlorotUniform(),
    # layer maxpool1
    maxpool1_pool_size=(2, 2),
    # layer conv2d2
    conv2d2_num_filters=32,
    conv2d2_filter_size=(5, 5),
    conv2d2_nonlinearity=lasagne.nonlinearities.rectify,
    # layer maxpool2
    maxpool2_pool_size=(2, 2),
    # dropout1
    dropout1_p=0.5,
    # dense
    dense_num_units=256,
    dense_nonlinearity=lasagne.nonlinearities.rectify,
    # dropout2
    dropout2_p=0.5,
    # output
    output_nonlinearity=lasagne.nonlinearities.softmax,
    output_num_units=10,
    # optimization method params
    update=nesterov_momentum,
    update_learning_rate=0.01,
    update_momentum=0.9,
    max_epochs=10,
    verbose=1,
    )
# Train the network
print("Start train")
nn = net1.fit(X_train, y_train)

print("Predict")
preds = net1.predict(X_test)

print("Print results:")
cm = confusion_matrix(y_test, preds)
plt.matshow(cm)
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()