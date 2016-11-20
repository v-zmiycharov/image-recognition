import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, os.pardir, "_DATA")
IMAGE_NET_DIR = os.path.join(DATA_DIR, "ImageNet")
IMAGES_DIR_NAME = "images"
TEST_DIR_NAME = "test"
CUSTOM_DIR = os.path.join(ROOT_DIR, os.pardir, "_CUSTOM")
BIN_DATA_DIR = os.path.join(CUSTOM_DIR, "_BIN_DATA")
MODEL_DATA_DIR = os.path.join(CUSTOM_DIR, "_MODEL")
