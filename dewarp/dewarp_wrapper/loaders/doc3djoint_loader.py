import torch
import numpy as np
import cv2
import hdf5storage as h5
import collections

from scipy import misc as m
from os.path import join as pjoin
from torch.utils import data


class doc3djointLoader(data.Dataset):
    def __init__(self, root, split='train', is_transform=False, img_size=512, bm_size=128):
        self.root = root
        self.split = split
        self.is_transform = is_transform
        self.img_size = img_size
        self.bm_size = bm_size
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
        wc_path = pjoin(self.root, 'wc', img_name + '.exr')
        bm_path = pjoin(self.root, 'bm', img_name + '.mat')
        recon_folder = 'chess48'
        recon_path = pjoin(self.root, 'recon',
                           img_name[:-4] + recon_folder + '0001.png')

        img = m.imread(img_path, mode='RGB')
        img = np.array(img, dtype=np.uint8)

        wc = cv2.imread(wc_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        wc = np.array(wc, dtype=np.float)

        bm = self.loadmat(bm_path)['bm']
        recon = m.imread(recon_path, mode='RGB')

        if self.is_transform:
            img, wc, bm, recon = self.transform(img, wc, bm, recon)

        return img, wc, bm, recon

    def loadmat(self, path):
        try:
            return h5.loadmat(path)
        except NotImplementedError:
            print(f'Couldn\'t read file: {path}')

    def transform(self, img, wc, bm, recon):
        img = m.imresize(img, (self.img_size[0], self.img_size[1]))
        if img.shape[2] == 4:
            img = img[:, :, :3]
        img = img[:, :, ::-1]  # RGB => BGR
        img = img.astype(np.float64)
        img = img.transpose(2, 0, 1)

        recon = m.imresize(recon, (self.bm_size[0], self.bm_size[1]))
        if recon.shape[2] == 4:
            recon = recon[:, :, :3]
        recon = recon[:, :, ::-1]  # RGB => BGR
        recon = recon.astype(np.float64)
        recon = recon.transpose(2, 0, 1)  # NHWC => NCHW

        msk = ((wc[:, :, 0] != 0) & (wc[:, :, 1] != 0)).astype(np.uint8) * 255
        xmx, xmn, ymx, ymn, zmx, zmn = 1.2539363, - \
            1.2442188, 1.2396319, -1.2289206, 0.6436657, -0.67187124
        wc[:, :, 0] = (wc[:, :, 0] - zmn) / (zmx - zmn)
        wc[:, :, 1] = (wc[:, :, 1] - ymn) / (ymx - ymn)
        wc[:, :, 2] = (wc[:, :, 2] - xmn) / (xmx - xmn)
        wc = cv2.bitwise_and(wc, wc, mask=msk)
        wc = m.imresize(wc, (self.img_size[0], self.img_size[1]))
        wc = wc.transpose(2, 0, 1)

        bm = bm.astype(float)
        xmx, xmn, ymx, ymn = 435.14545757153445, 13.410177297916455, 435.3297804574046, 14.194541402379988
        bm[:, :, 0] = (bm[:, :, 0] - xmn) / (xmx - xmn)
        bm[:, :, 1] = (bm[:, :, 1] - ymn) / (ymx - ymn)
        bm = (bm - 0.5) * 2

        bm0 = cv2.resize(bm[:, :, 0], (self.bm_size[0], self.bm_size[1]))
        bm1 = cv2.resize(bm[:, :, 1], (self.bm_size[0], self.bm_size[1]))

        bm = np.stack([bm0, bm1], axis=-1)

        # Load tensor
        img = torch.from_numpy(img).float()
        wc = torch.from_numpy(wc).float()
        bm = torch.from_numpy(bm).float()
        recon = torch.from_numpy(recon).float()

        return img, wc, bm, recon
