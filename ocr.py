import cv2
import pytesseract as tess
import torch
import torch.nn as nn
from utils.img_proc import ImageProcessing
from utils.CRAFT import *

img_path = 'resources/Samples/good-anonymized/ams-302910.jpg'

img = cv2.imread(img_path)


while True:
    cv2.imshow('', img)
    print(tess.image_to_string(img))
    key = cv2.waitKey(33)
    if key == 27:
        break
