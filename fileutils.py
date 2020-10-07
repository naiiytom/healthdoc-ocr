from os.path import join as pjoin
import shutil
import time

ls_file = 'D:/doc3d-dataset/listfile.txt'
SRC_PATH = 'D:/doc3d-dataset'
DST_PATH = 'C:/Users/yuttapichai.lam/dev-environment/doc3d'

files = open(ls_file, 'r')
start = time.time()
for file in files:
    name = file[:-5]
    print(f'File: {name}')
    shutil.copy(pjoin(SRC_PATH, 'img', name + '.png'),
                pjoin(DST_PATH, 'img', name + '.png'))
    shutil.copy(pjoin(SRC_PATH, 'wc', name + '.exr'),
                pjoin(DST_PATH, 'wc', name + '.exr'))
    shutil.copy(pjoin(SRC_PATH, 'bm', name + '.mat'),
                pjoin(DST_PATH, 'bm', name + '.mat'))
    shutil.copy(pjoin(SRC_PATH, 'recon', name[:-4] + 'chess480001.png'), pjoin(
        DST_PATH, 'recon', name[:-4] + 'chess480001.png'))
finish = time.time() - start
print(f'Finished in {finish} seconds')
