import torch
import numpy as np
import cv2
import hdf5storage as h5
import collections

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
        img = img.transpose(2, 0, 1)

        # msk = ((norm[:, :, 0] != 0) & (norm[:, :, 1] != 0)).astype(np.uint8) * 255
        # xmx, xmn, ymx, ymn, zmx, zmn = 1.2539363, - \
        #     1.2442188, 1.2396319, -1.2289206, 0.6436657, -0.67187124
        # norm[:, :, 0] = (norm[:, :, 0] - zmn) / (zmx - zmn)
        # norm[:, :, 1] = (norm[:, :, 1] - ymn) / (ymx - ymn)
        # norm[:, :, 2] = (norm[:, :, 2] - xmn) / (xmx - xmn)
        # norm = cv2.bitwise_and(norm, norm, mask=msk)
        norm = m.imresize(norm, (self.img_size[0], self.img_size[1]))
        norm = norm.astype(float)
        norm = norm.transpose(2, 0, 1)

        # Load tensor
        img = torch.from_numpy(img).float()
        norm = torch.from_numpy(norm).float()


        return img, norm


class doc3dSELoader(data.Dataset):
    pass
