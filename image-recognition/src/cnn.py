import os
import numpy as np
import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from PIL import Image
import definitions

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def load_dataset():
    IMAGE_SIZE = 28
    X_train = []
    X_test = []
    y_train = []
    y_test = []

    allowed_labels = ["alligator", "antelope"]

    for label in allowed_labels:
        train_dir = os.path.join(definitions.IMAGE_NET_DIR, label, definitions.IMAGES_DIR_NAME)
        test_dir = os.path.join(definitions.IMAGE_NET_DIR, label, definitions.TEST_DIR_NAME)
        for filename in os.listdir(train_dir):
            image = Image.open(os.path.join(train_dir, filename))
            image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
            X_train.append(np.array(image))
            y_train.append(label)
        for filename in os.listdir(test_dir):
            image = Image.open(os.path.join(test_dir, filename))
            image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
            X_test.append(np.array(image))
            y_test.append(label)

    return X_train, np.array(y_train), X_test, np.array(y_test)

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
    input_shape=(None, 1, 28, 28),
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
nn = net1.fit(X_train, y_train)

preds = net1.predict(X_test)

cm = confusion_matrix(y_test, preds)
plt.matshow(cm)
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()