import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data

import grad_loss
from models import get_model
from loaders import get_loader


def train():
    refine_model_name = 'unetnc'
    data_path = 'C:/Users/yuttapichai.lam/dev-environment/dataset'
    data_loader = get_loader('doc3drefine')
    t_loader = data_loader(data_path, is_transform=True,
                           img_size=(256, 256), bm_size=(128, 128))
    v_loader = data_loader(data_path, split='val',
                           is_transform=True, img_size=(256, 256), bm_size=(128, 128))

    model = get_model(refine_model_name, n_classes=3, in_channel=6)


if __name__ == "__main__":
    train()
