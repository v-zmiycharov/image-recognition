import src.main.models as models

IMAGE_SIZE = 224
IMAGE_DEPTH = 3

NUM_CLASSES = 10

EPOCHS = 25
LEARN_RATE = 0.1
MOMENTUM = 0.9
BATCH_SIZE = 100

CROSS_VALIDATION_ENABLED = False
N_FOLD_CROSS_VALIDATION = 5
IMAGES_IN_BATCH = 1000

TRAIN_BATCH_PREFIX = 'train_batch_'
TEST_BATCH_PREFIX = 'test_batch_'
METADATA_FILENAME = '_metadata.bin'

MODEL_JSON_FILENAME = '_model.json'
MODEL_WEIGHTS_FILENAME = '_model.h5'

NESTEROV = True

model_function = models.create_vgg16_model
