"""

"""

import cv2
import numpy as np
from skimage import io


def load_image(img_path):
    img = io.imread(img_path)

    if img.shape[0] == 2:
        img = img[0]
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:
        img = img[:, :, 3]

    img = np.array(img)

    return img


def greetings(name='Tom'):
    print(f'Hello, {name}')
