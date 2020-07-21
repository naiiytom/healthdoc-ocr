import torch
from torch import dtype
import torch.nn.functional as F
import numpy as np
import cv2
import hdf5storage as h5
import collections
import matplotlib.pyplot as plt

from scipy import misc as m
from os.path import join as pjoin
from torch.utils import data


class doc3dNELoader(data.Dataset):
    def __init__(self, root, split='train', is_transform=False, img_size=512):
        self.root = root
        self.split = split
        self.is_transform = is_transform
        self.img_size = img_size
        self.files = collections.defaultdict(list)

        for split in ['train', 'val']:
            path = pjoin(self.root, split + '.txt')
            file_list = tuple(open(path, 'r'))
            file_list = [id_.rstrip() for id_ in file_list]
            self.files[split] = file_list

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        img_name = self.files[self.split][index]
        img_name = img_name[:-4]
        img_path = pjoin(self.root, 'img', img_name + '.png')
        norm_path = pjoin(self.root, 'norm', img_name + '.exr')

        img = m.imread(img_path, mode='RGB')
        img = np.array(img, dtype=np.uint8)

        norm = cv2.imread(norm_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        norm = cv2.cvtColor(norm, cv2.COLOR_RGB2BGR)
        norm = np.array(norm, dtype=np.float)

        if self.is_transform:
            img, norm = self.transform(img, norm)

        return img, norm

    def transform(self, img, norm):
        img = m.imresize(img, (self.img_size[0], self.img_size[1]))
        if img.shape[-1] == 4:
            img = img[:, :, :3]
        img = img[:, :, ::-1]  # RGB => BGR
        img = img.astype(float) / 255.0
        img = img.transpose((2, 0, 1))

        # msk = ((norm[:, :, 0] != 0) & (norm[:, :, 1] != 0)).astype(np.uint8) * 255
        # zmx, zmn = np.max(norm[:, :, 0]), np.min(norm[:, :, 0])
        # ymx, ymn = np.max(norm[:, :, 1]), np.min(norm[:, :, 1])
        # xmx, xmn = np.max(norm[:, :, 2]), np.min(norm[:, :, 2])

        # norm[:, :, 0] = (norm[:, :, 0] - zmn) / (zmx - zmn)
        # norm[:, :, 1] = (norm[:, :, 1] - ymn) / (ymx - ymn)
        # norm[:, :, 2] = (norm[:, :, 2] - xmn) / (xmx - xmn)
        # norm = cv2.bitwise_and(norm, norm, mask=msk)
        norm = cv2.resize(norm, (self.img_size[0], self.img_size[1]), interpolation=cv2.INTER_NEAREST)
        norm = norm.astype(float)
        norm = norm.transpose((2, 0, 1))

        # Load tensor
        img = torch.from_numpy(img).float()
        norm = torch.from_numpy(norm).float()

        return img, norm


class doc3dSELoader(data.Dataset):
    pass


if __name__ == "__main__":
    ROOT = 'C:/Users/yuttapichai.lam/dev-environment/dataset/'
    FNAME = '1_1_2-pm_Page_144-Tv60001'
    img_path = ROOT + 'img/' + FNAME + '.png'
    img = m.imread(img_path, mode='RGB')
    # img = np.array(img, dtype=np.uint8)
    norm_path = ROOT + 'norm/' + FNAME + '.exr'
    norm = cv2.imread(norm_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    norm = cv2.cvtColor(norm, cv2.COLOR_RGB2BGR)
    # norm = np.array(norm, dtype=np.float)
    bm_path = ROOT + 'bm/' + FNAME + '.mat'
    bm = h5.loadmat(bm_path)['bm']
    # bm = np.array(bm, dtype=np.float)

    img = img[:, :, ::-1]
    im = img.transpose((2, 0, 1))
    im = im.astype(np.float) / 255.0
    im = np.expand_dims(im, 0)
    im = torch.from_numpy(im).float()


    bm = bm / np.array([448, 448])
    bm = (bm - 0.5) * 2
    bm0 = bm[:, :, 0]
    bm1 = bm[:, :, 1]
    bm = np.stack([bm0, bm1], axis=-1)
    # bm = np.reshape(bm, (1, 448, 448, 2))
    bm = np.expand_dims(bm, 0)
    bm = torch.from_numpy(bm).float()

    # msk = ((norm[:, :, 0] != 0) & (norm[:, :, 1] != 0)).astype(np.uint8) * 255
    # zmx, zmn = np.max(norm[:, :, 0]), np.min(norm[:, :, 0])
    # ymx, ymn = np.max(norm[:, :, 1]), np.min(norm[:, :, 1])
    # xmx, xmn = np.max(norm[:, :, 2]), np.min(norm[:, :, 2])

    # norm[:, :, 0] = (norm[:, :, 0] - zmn) / (zmx - zmn)
    # norm[:, :, 1] = (norm[:, :, 1] - ymn) / (ymx - ymn)
    # norm[:, :, 2] = (norm[:, :, 2] - xmn) / (xmx - xmn)
    # norm = cv2.bitwise_and(norm, norm, mask=msk)
    norm = norm.astype(np.float)
    nm = norm.transpose((2, 0, 1))
    nm = np.expand_dims(nm, 0)
    nm = torch.from_numpy(nm).float()

    imorg = F.grid_sample(im, bm, mode='bilinear', align_corners=True)
    imorg = imorg[0].numpy().transpose((1, 2, 0))
    normorg = F.grid_sample(nm, bm, mode='bilinear', align_corners=True)
    normorg = normorg[0].numpy().transpose((1, 2, 0))

    _, ax = plt.subplots(2, 2)
    ax[0][0].imshow(img)
    ax[0][1].imshow(norm)
    ax[1][0].imshow(imorg)
    ax[1][1].imshow(normorg)
    plt.show()
    # img = img.transpose(2, 0, 1)
    # norm = norm.transpose(2, 0, 1)
    # print(norm)
    # print(np.max(img[:, :, 0]))
    # print(np.max(img[:, :, 1]))
    # print(np.max(img[:, :, 2]))

    # print(np.max(norm[:, :, 0]))
    # print(np.max(norm[:, :, 1]))
    # print(np.max(norm[:, :, 2]))
    # print(np.min(norm[:, :, 0]))
    # print(np.min(norm[:, :, 1]))
    # print(np.min(norm[:, :, 2]))

    # print(np.max(img))
    # print(np.min(img))
    # print(np.max(norm))
    # print(np.min(norm))
