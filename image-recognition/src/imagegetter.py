import definitions
import os
import requests
from time import localtime, strftime
import tarfile
from multiprocessing.dummy import Pool as ThreadPool
import random
from src.image_net import IMAGES

def generate_download_url(synset_id):
    return "http://image-net.org/download/synset?wnid={0}&username=vzmiycharov&accesskey=9fbd7b85ed46b46b2aed80e6335ac9ee5f5b9463&release=latest&src=stanford" \
                .format(synset_id)

def download_file(url, file_path):
    r = requests.get(url, stream=True)
    with open(file_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024*1024*5):
            if chunk:
                f.write(chunk)

def extract_tar_file(file_path):
    if os.path.isfile(file_path):
        with tarfile.open(file_path) as tar:
            folder_path = os.path.dirname(file_path)
            tar.extractall(path=folder_path)

def create_test_set(images_dir, test_dir):
    images_count = len(os.listdir(images_dir))
    test_images_count = images_count // 20

    random_test_images = random.sample(os.listdir(images_dir), test_images_count)
    for file_name in random_test_images:
        os.rename(os.path.join(images_dir, file_name), os.path.join(test_dir, file_name))


def get_metadata():
    for label in IMAGES:
        animal = label[1]
        synset_id = label[2]
        animal_dir = os.path.join(definitions.IMAGE_NET_DIR, animal)
        print("{2}: {0} ({1}):".format(animal, synset_id, strftime("%Y-%m-%d %H:%M:%S", localtime())))
        if not os.path.exists(animal_dir):
            os.makedirs(animal_dir)
        images_dir = os.path.join(animal_dir, definitions.IMAGES_DIR_NAME)
        if not os.path.exists(images_dir):
            os.makedirs(images_dir)

        test_dir = os.path.join(animal_dir, definitions.TEST_DIR_NAME)
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)

        url = generate_download_url(synset_id)
        file_name = synset_id + ".tar"
        file_path = os.path.join(images_dir, file_name)

        yield (animal, synset_id, animal_dir, images_dir, test_dir, file_name, file_path, url)

def process_animal(tupple):
    (animal, synset_id, animal_dir, images_dir, test_dir, file_name, file_path, url) = tupple
    if len(os.listdir(images_dir)) == 0:
        print("{2}: Downloading {0} ({1}) ...".format(animal, synset_id, strftime("%Y-%m-%d %H:%M:%S", localtime())))
        download_file(url, file_path)

    if os.path.isfile(file_path):
        print("{2}: Extracting {0} ({1}) ...".format(animal, synset_id, strftime("%Y-%m-%d %H:%M:%S", localtime())))
        extract_tar_file(file_path)
        print("{2}: Deleting {0} ({1}) ...".format(animal, synset_id, strftime("%Y-%m-%d %H:%M:%S", localtime())))
        os.remove(file_path)

    if len(os.listdir(test_dir)) == 0:
        print("{2}: Creating test set {0} ({1}) ...".format(animal, synset_id, strftime("%Y-%m-%d %H:%M:%S", localtime())))
        create_test_set(images_dir, test_dir)


pool = ThreadPool(4)
pool.map(process_animal, get_metadata())



