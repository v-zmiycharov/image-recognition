import definitions
import os
import requests
from time import localtime, strftime
import tarfile

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

with open(os.path.join(definitions.IMAGE_NET_DIR, "list.txt")) as list_file:
    for line in list_file:
        splitted = line.rstrip().split(" - ")
        if len(splitted) == 2:
            animal = splitted[0]
            synset_id = splitted[1]
            animal_dir = os.path.join(definitions.IMAGE_NET_DIR, animal)
            print("{2}: {0} ({1}):".format(animal, synset_id, strftime("%Y-%m-%d %H:%M:%S", localtime())))
            if not os.path.exists(animal_dir):
                os.makedirs(animal_dir)
            animal_dir = os.path.join(animal_dir, "images")
            if not os.path.exists(animal_dir):
                os.makedirs(animal_dir)

            url = generate_download_url(synset_id)
            file_path = os.path.join(animal_dir, synset_id + ".tar")

            if len(os.listdir(animal_dir)) == 0:
                print("{2}: Downloading {0} ({1}) ...".format(animal, synset_id, strftime("%Y-%m-%d %H:%M:%S", localtime())))
                download_file(url, file_path)

            if os.path.isfile(file_path):
                print("{2}: Extracting {0} ({1}) ...".format(animal, synset_id, strftime("%Y-%m-%d %H:%M:%S", localtime())))
                extract_tar_file(file_path)
                print("{2}: Deleting {0} ({1}) ...".format(animal, synset_id, strftime("%Y-%m-%d %H:%M:%S", localtime())))
                os.remove(file_path)


