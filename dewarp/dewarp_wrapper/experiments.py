import hdf5storage as h5
import numpy as np
import scipy.misc as m
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt

import cv2

ROOT = 'D:/doc3d-dataset/'
FNAME = '1_1_3-pp_Page_069-tCY0001'
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
alb_path = ROOT + 'alb/' + FNAME + '.png'
alb = m.imread(alb_path, mode='RGB')

# img = img[:, :, ::-1]
im = img.transpose((2, 0, 1))
im = im.astype(np.float) / 255.0
im = np.expand_dims(im, 0)
im = torch.from_numpy(im).float()

al = alb.transpose((2, 0, 1))
al = al.astype(np.float) / 255.0
al = np.expand_dims(al, 0)
al = torch.from_numpy(al).float()


bm = bm / np.array([448, 448])
bm = (bm - 0.5) * 2
bm0 = cv2.blur(bm[:, :, 0], (5, 5))
bm1 = cv2.blur(bm[:, :, 1], (5, 5))
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
alorg = F.grid_sample(al, bm, mode='bilinear', align_corners=True)
alorg = alorg[0].numpy().transpose((1, 2, 0))
normorg = F.grid_sample(nm, bm, mode='bilinear', align_corners=True)
normorg = normorg[0].numpy().transpose((1, 2, 0))
shade = cv2.cvtColor(normorg, cv2.COLOR_RGB2GRAY)
# kernel = np.ones((11, 11), np.float32)/121
# shade = cv2.filter2D(shade, -1, kernel)
# shade = cv2.filter2D(shade, -1, kernel)
# shade = cv2.filter2D(shade, -1, kernel
shade = cv2.medianBlur(shade, 5)
shade = np.stack([shade, shade, shade], axis=-1)
# gray = cv2.cvtColor(imorg, cv2.COLOR_BGR2GRAY)
print(imorg.shape)
print(normorg.shape)
print(shade.shape)
# imorg = (imorg * 255).astype(np.uint8)
res = np.add(imorg, shade)
# res = cv2.bitwise_and(imorg, imorg, mask=shade)
# res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
res = np.array(res)
print(res.shape)

_, ax = plt.subplots(4, 2)
ax[0][0].imshow(img)
ax[0][1].imshow(norm)
ax[1][0].imshow(imorg)
ax[1][1].imshow(normorg)
ax[2][0].imshow(shade, cmap='gray')
ax[2][1].imshow(res)
ax[3][0].imshow(alb)
ax[3][1].imshow(alorg)
plt.show()
