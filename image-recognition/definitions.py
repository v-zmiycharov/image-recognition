import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, os.pardir, "_DATA")
IMAGE_NET_DIR = os.path.join(DATA_DIR, "ImageNet")
MODELS_DIR = os.path.join(ROOT_DIR, os.pardir, "_MODELS")
