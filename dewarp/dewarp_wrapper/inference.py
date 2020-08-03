#test end to end benchmark data test
import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable
from tqdm import tqdm

import cv2
from loaders import get_loader
from models import get_model
from utils import convert_state_dict

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def unwarp(img, bm):
    w,h=img.shape[0],img.shape[1]
    bm = bm.transpose(1, 2).transpose(2, 3).detach().cpu().numpy()[0,:,:,:]
    bm0=cv2.blur(bm[:,:,0],(3,3))
    bm1=cv2.blur(bm[:,:,1],(3,3))
    bm0=cv2.resize(bm0,(h,w))
    bm1=cv2.resize(bm1,(h,w))
    bm=np.stack([bm0,bm1],axis=-1)
    bm=np.expand_dims(bm,0)
    bm=torch.from_numpy(bm).double()
    

    img = img.astype(float) / 255.0
    img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, 0)
    img = torch.from_numpy(img).double()

    res = F.grid_sample(input=img, grid=bm, align_corners=True)
    res = res[0].numpy().transpose((1, 2, 0))

    return res


def test(args,img_path,fname):
    wc_model_file_name = os.path.split(args.wc_model_path)[1]
    wc_model_name = wc_model_file_name[:wc_model_file_name.find('_')]

    bm_model_file_name = os.path.split(args.bm_model_path)[1]
    bm_model_name = bm_model_file_name[:bm_model_file_name.find('_')]

    wc_n_classes = 3
    bm_n_classes = 2

    wc_img_size=(256,256)
    bm_img_size=(128,128)

    # Setup image
    print("Read Input Image from : {}".format(img_path))
    imgorg = cv2.imread(img_path)
    imgorg = cv2.cvtColor(imgorg, cv2.COLOR_BGR2RGB)
    img = cv2.resize(imgorg, wc_img_size)
    img = img[:, :, ::-1]
    img = img.astype(float) / 255.0
    img = img.transpose(2, 0, 1) # NHWC -> NCHW
    img = np.expand_dims(img, 0)
    img = torch.from_numpy(img).float()

    # Predict
    htan = nn.Hardtanh(0,1.0)
    wc_model = get_model(wc_model_name, wc_n_classes, in_channels=3)
    if DEVICE.type == 'cpu':
        wc_state = convert_state_dict(torch.load(args.wc_model_path, map_location='cpu')['model_state'])
    else:
        wc_state = convert_state_dict(torch.load(args.wc_model_path)['model_state'])
    wc_model.load_state_dict(wc_state)
    wc_model.eval()
    bm_model = get_model(bm_model_name, bm_n_classes, in_channels=3)
    if DEVICE.type == 'cpu':
        bm_state = convert_state_dict(torch.load(args.bm_model_path, map_location='cpu')['model_state'])
    else:
        bm_state = convert_state_dict(torch.load(args.bm_model_path)['model_state'])
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
        pred_wc = htan(wc_outputs)
        bm_input=F.interpolate(pred_wc, bm_img_size, mode='bilinear', align_corners=True)
        outputs_bm = bm_model(bm_input)

        # call unwarp
        wc = pred_wc[0].cpu().detach().numpy().transpose((1, 2, 0))
        wc = cv2.resize(wc, (imgorg.shape[1], imgorg.shape[0]))
        mask = (wc * 255).astype(np.uint8)
        _, binary_im = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
        # binary_im = binary_im.astype(np.float32) / 255.0
        out = cv2.bitwise_and(imgorg, binary_im)
        bm = outputs_bm[0].cpu().detach().numpy().transpose((1, 2, 0))
        bm = cv2.resize(bm, (imgorg.shape[1], imgorg.shape[0]))
        uwpred=unwarp(imgorg, outputs_bm)

    if args.show:
        f1, axarr1 = plt.subplots(1, 6)
        axarr1[0].imshow(imgorg)
        axarr1[1].imshow(wc)
        axarr1[2].imshow(bm[:, :, 0])
        axarr1[3].imshow(uwpred)
        axarr1[4].imshow(binary_im, cmap='gray')
        axarr1[5].imshow(out)
        plt.show()

    # Save the output
    mask_out=os.path.join(args.out_path, fname[:-4] +'___.png')
    img_out=os.path.join(args.out_path,fname)
    wc_out=os.path.join(args.out_path, fname[:-4] + '__.png')
    cv2.imwrite(img_out,uwpred[:,:,::-1]*255)
    cv2.imwrite(wc_out, wc[:, :, ::-1]*255)
    cv2.imwrite(mask_out, out[:, :, ::-1])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Params')
    parser.add_argument('--wc_model_path', nargs='?', type=str, default='',
                        help='Path to the saved wc model')
    parser.add_argument('--bm_model_path', nargs='?', type=str, default='',
                        help='Path to the saved bm model')
    parser.add_argument('--img_path', nargs='?', type=str, default='./eval/inp/',
                        help='Path of the input image')
    parser.add_argument('--out_path', nargs='?', type=str, default='./eval/out/',
                        help='Path of the output unwarped image')
    parser.add_argument('--show', dest='show', action='store_true',
                        help='Show the input image and output unwarped')
    parser.set_defaults(show=False)
    args = parser.parse_args()
    for fname in os.listdir(args.img_path):
        if '.jpg' in fname or '.JPG' in fname or '.png' in fname:
            img_path=os.path.join( args.img_path,fname)
            test(args,img_path,fname)


# python infer.py --wc_model_path ./eval/models/unetnc_doc3d.pkl --bm_model_path ./eval/models/dnetccnl_doc3d.pkl --show
