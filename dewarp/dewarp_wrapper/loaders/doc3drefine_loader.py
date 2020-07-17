import torch
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

        msk = ((norm[:, :, 0] != 0) & (norm[:, :, 1] != 0)).astype(np.uint8) * 255
        zmx, zmn = np.max(norm[:, :, 0]), np.min(norm[:, :, 0])
        ymx, ymn = np.max(norm[:, :, 1]), np.min(norm[:, :, 1])
        xmx, xmn = np.max(norm[:, :, 2]), np.min(norm[:, :, 2])

        norm[:, :, 0] = (norm[:, :, 0] - zmn) / (zmx - zmn)
        norm[:, :, 1] = (norm[:, :, 1] - ymn) / (ymx - ymn)
        norm[:, :, 2] = (norm[:, :, 2] - xmn) / (xmx - xmn)
        norm = cv2.bitwise_and(norm, norm, mask=msk)
        norm = m.imresize(norm, (self.img_size[0], self.img_size[1]))
        norm = norm.astype(float)
        norm = norm.transpose(2, 0, 1)

        # Load tensor
        img = torch.from_numpy(img).float()
        norm = torch.from_numpy(norm).float()


        return img, norm


class doc3dSELoader(data.Dataset):
    pass




if __name__ == "__main__":
    ROOT = 'C:/Users/yuttapichai.lam/dev-environment/dataset/'
    img_path = ROOT + 'img/1_1_1-pr_Page_141-PZU0001.png'
    img = m.imread(img_path, mode='RGB')
    img = np.array(img, dtype=np.uint8)
    norm_path = ROOT + 'norm/1_1_1-pr_Page_141-PZU0001.exr'
    norm = cv2.imread(norm_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH) * 255
    norm = np.array(norm, dtype=np.float)
    
    # img = img[:, :, ::-1]
    img = (img/255.0).astype(np.float)
    # img = img.transpose(2, 0, 1)

    # msk = ((norm[:, :, 0] != 0) & (norm[:, :, 1] != 0)).astype(np.uint8) * 255
    # zmx, zmn = np.max(norm[:, :, 0]), np.min(norm[:, :, 0])
    # ymx, ymn = np.max(norm[:, :, 1]), np.min(norm[:, :, 1])
    # xmx, xmn = np.max(norm[:, :, 2]), np.min(norm[:, :, 2])

    # norm[:, :, 0] = (norm[:, :, 0] - zmn) / (zmx - zmn)
    # norm[:, :, 1] = (norm[:, :, 1] - ymn) / (ymx - ymn)
    # norm[:, :, 2] = (norm[:, :, 2] - xmn) / (xmx - xmn)
    # norm = cv2.bitwise_and(norm, norm, mask=msk)
    norm = norm.astype(np.float) / 255.0
    # norm = norm.transpose(2, 0, 1)

    _, ax = plt.subplots(2, 1)
    ax[0].imshow(img)
    ax[1].imshow(norm)
    plt.show()
    img = img.transpose(2, 0, 1)
    norm = norm.transpose(2, 0, 1)
    print(norm)
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
