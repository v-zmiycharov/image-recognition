import definitions
import os
import requests
from time import gmtime, strftime

def generate_download_url(synset_id):
    return "http://image-net.org/download/synset?wnid={0}&username=vzmiycharov&accesskey=9fbd7b85ed46b46b2aed80e6335ac9ee5f5b9463&release=latest&src=stanford" \
                .format(synset_id)

def download_file(url, file_path):
    r = requests.get(url, stream=True)
    with open(file_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024*1024*5):
            if chunk:
                f.write(chunk)

def download_archives():
    with open(os.path.join(definitions.IMAGE_NET_DIR, "list.txt")) as list_file:
        for line in list_file:
            splitted = line.rstrip().split(" - ")
            if len(splitted) == 2:
                animal = splitted[0]
                synset_id = splitted[1]
                animal_dir = os.path.join(definitions.IMAGE_NET_DIR, animal)
                if not os.path.exists(animal_dir):
                    os.makedirs(animal_dir)

                url = generate_download_url(synset_id)
                file_path = os.path.join(animal_dir, synset_id + ".tar")
                print("{2}: Downloading {0} ({1}) ...".format(animal, synset_id, strftime("%Y-%m-%d %H:%M:%S", gmtime())))
                download_file(url, file_path)
                print("{2}: Download {0} ({1}) finished".format(animal, synset_id, strftime("%Y-%m-%d %H:%M:%S", gmtime())))

download_archives()
