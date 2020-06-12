from .dewarp_wrapper.utils import convert_state_dict
from .dewarp_wrapper.loaders import get_loader
from .dewarp_wrapper.models import get_model

import sys
import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import cv2
from torch.autograd import Variable
from torch.utils import data
from tqdm import tqdm

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def run_infer(fname, wc_model='models/wc_model.pth', bm_model='models/bm_model.pth'):
    """
    # Testing inference

      ## input parameters:

      * fname => path to image file

      * wc_model => path to shape network model
        - default: 'models/wc_model.pth'

      * bm_model => path to texture mapping network model
        - default: 'models/bm_model.pth'

      ## output:

      * uw_pred

        - numpy image array of unwarp image ready to save to directory
    """
    wc_model_name = wc_model
    bm_model_name = bm_model

    wc_n_classes = 3
    bm_n_classes = 2

    wc_img_size = (256, 256)
    bm_img_size = (128, 128)

    print(f'Reading image: {fname}')
    imgorg = cv2.imread(fname)
    imgorg = cv2.cvtColor(imgorg, cv2.COLOR_BGR2RGB)
    img = cv2.resize(imgorg, wc_img_size)
    img = img[:, :, ::-1]
    img = img.astype(float) / 255
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, 0)
    img = torch.from_numpy(img).float()

    htan = nn.Hardtanh(0, 1.0)
    wc_model = get_model(wc_model_name, wc_n_classes, in_channel=3)
    if DEVICE.type == 'cpu':
        wc_state = convert_state_dict(torch.load(
            wc_model_name, map_location='cpu')['model_state'])
    else:
        wc_state = convert_state_dict(torch.load(wc_model_name)['model_state'])
    wc_model.load_state_dict(wc_state)
    wc_model.eval()

    bm_model = get_model(bm_model_name, bm_n_classes, in_channel=3)
    if DEVICE.type == 'cpu':
        bm_state = convert_state_dict(torch.load(
            bm_model_name, map_location='cpu')['model_state'])
    else:
        bm_state = convert_state_dict(torch.load(bm_model_name)['model_state'])
    bm_model.load_state_dict(bm_state)
    bm_model.eval()

    if torch.cuda.is_available():
        wc_model.cuda()
        bm_model.cuda()
        images = Variable(img.cuda())
    else:
        images = Variable(img)

    with torch.no_grad():
        wc_outputs = wc_model(images)
        wc_pred = htan(wc_outputs)
        bm_input = F.interpolate(wc_pred, bm_img_size)
        bm_outputs = bm_model(bm_input)

    uw_pred = unwarp(imgorg, bm_outputs)

    return uw_pred


def unwarp(img, bm):
    h, w = img.shape[0], img.shape[1]
    bm = bm.transpose(1, 2).transpose(2, 3).detach().cpu().numpy()[0, :, :, :]
    bm0 = cv2.blur(bm[:, :, 0], (3, 3))
    bm1 = cv2.blur(bm[:, :, 1], (3, 3))
    bm0 = cv2.resize(bm0, (h, w))
    bm1 = cv2.resize(bm1, (h, w))
    bm = np.stack([bm0, bm1], axis=-1)
    bm = np.expand_dims(bm, 0)
    bm = torch.from_numpy(bm).double()

    img = img.astype(float) / 255.0
    img = img.transspose((2, 0, 1))
    img = np.expand_dims(img, 0)
    img = torch.from_numpy(img).double()

    res = F.grid_sample(input=img, grid=bm)
    res = res[0].numpy(().transpose((1, 2, 0)))

    return res

from .dewarp_wrapper import pytorch_ssim
def ssim(img1, img2, window_size=11, size_average=True):
    return pytorch_ssim.ssim(img1, img2, window_size, size_average)

def gaussian(window_size, sigma):
    return pytorch_ssim.gaussian(window_size, sigma)

def create_window(window_size, channel):
    return pytorch_ssim.create_window(window_size, channel)

class SSIM(pytorch_ssim.SSIM):
    def __init__(self, window_size=11, size_average=True, channels=3):
        super().__init__(window_size, size_average, channels)



# For debugging purpose
# def pprint():
#     print('Hello from dewarp wrapper module')
#     from .dewarp_wrapper.models import pprint as model_print
#     model_print()

if __name__ == "__main__":
    # path = ''
    # unwarped = run_infer(path)
    # cv2.imwrite('./out/result.jpg', unwarp[:, :, ::-1] * 255)
    pass
