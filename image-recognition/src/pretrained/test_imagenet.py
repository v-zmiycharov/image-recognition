from src.pretrained.resnet50 import ResNet50
from keras.preprocessing import image
from src.pretrained.imagenet_utils import preprocess_input, decode_predictions
import os
import numpy as np
import definitions


if __name__ == '__main__':
    model = ResNet50(weights='imagenet')

    img_path = os.path.join(definitions.IMAGE_NET_DIR, 'alligator/n01698434_154.JPEG')
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    print('Predicted:', decode_predictions(preds))