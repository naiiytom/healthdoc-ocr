import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.cuda import device_count
from torch.functional import norm
from torch.utils import data
from tqdm import tqdm

import cv2
from loaders import get_loader
from models import get_model
from utils import convert_state_dict


def train(n_epoch=50, batch_size=32, resume=False, tboard=False, ne_path=''):
    model_name = 'unetnc'

    # Setup dataloader
    data_path = 'C:/Users/yuttapichai.lam/dev-environment/doc3d'
    data_loader = get_loader('se_refine')
    t_loader = data_loader(data_path, is_transform=True,
                           img_size=(256, 256))
    v_loader = data_loader(data_path, split='val',
                           is_transform=True, img_size=(256, 256))

    trainloader = data.DataLoader(
        t_loader, batch_size=batch_size, num_workers=8, shuffle=True)
    valloader = data.DataLoader(v_loader, batch_size=batch_size, num_workers=8)

    # Load models
    print('Loading')
    model = get_model(model_name, n_classes=3, in_channels=6)
    model = torch.nn.DataParallel(
        model, device_ids=range(torch.cuda.device_count()))
    model.cuda()

    # print(model)
    # print(len(list(model.parameters)))

    # Setup optimizer and learning rate reduction
    print('Setting optimizer')
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-4, weight_decay=5e-4, amsgrad=True)
    schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    # Set Activation function
    htan = nn.Hardtanh(-1.0, 1.0)
    # sigmoid = nn.Sigmoid()

    # Setup losses
    MSE = nn.MSELoss()
    loss_fn = nn.L1Loss()

    epoch_start = 0
    global_step = 0

    if tboard:
        writer = SummaryWriter(comment='Refinement_SE')
    if resume:
        print('Resume from previous state')
        ne_chkpnt = torch.load(ne_path)
        model.load_state_dict(ne_chkpnt['model_state'])
        epoch_start = ne_chkpnt['epoch']
        optimizer.load_state_dict(ne_chkpnt['optimizer_state'])

    best_val_mse = 9999999.0
    print(f'Start from epoch {epoch_start} of {n_epoch}')
    print('Starting')
    for epoch in range(epoch_start, n_epoch):
        print(f'Epoch: {epoch}')
        # Loss initialization
        avg_loss = 0.0
        avg_l1loss = 0.0
        train_mse = 0.0

        # Start training
        model.train()

        print('Training')

        for i, (imgs, shades) in enumerate(trainloader):
            images = Variable(imgs.cuda())
            shade_labels = Variable(shades.cuda())

            optimizer.zero_grad()

            # Train SE network
            se_out = model(images)
            # ne_out = F.interpolate(ne_out, size=(
            #     256, 256), mode='bilinear', align_corners=True)
            # ne_pred = sigmoid(ne_out)
            se_pred = htan(se_out)

            # im = ne_pred.cpu().detach().numpy()
            # # print(im.shape)
            # im = im.transpose(0, 2, 3, 1)
            # print(im[0])
            # plt.imshow(im[0])
            # plt.show()

            l1_loss = loss_fn(se_pred, shade_labels)
            mse = MSE(se_pred, shade_labels)

            # SE Loss
            avg_l1loss += float(l1_loss)
            train_mse += float(mse)
            avg_loss += float(l1_loss)

            global_step += 1
            if (i+1) % 20 == 0:
                print(
                    f'Epoch[{epoch}/{n_epoch}] Batch[{i+1}/{len(trainloader)}] Loss: {avg_loss/(i+1):.6f} MSE: {train_mse/(i+1)}')

            if tboard and (i+1) % 20 == 0:
                writer.add_scalars('Train', {
                    'L1_Loss/train': avg_loss/(i+1),
                    'MSE_Loss/train': train_mse/(i+1)}, global_step)
            # mse.backward()
            l1_loss.backward()
            optimizer.step()

        len_trainset = len(trainloader)
        loss = avg_loss/len_trainset
        train_mse = train_mse/len_trainset
        train_losses = [loss, train_mse]
        print(f'SE L1 loss: {train_losses[0]} SE MSE: {train_losses[1]}')
        model.eval()
        val_l1 = 0.0
        val_mse = 0.0

        print('Validating')

        for i_val, (imgs_val, shades_val) in tqdm(enumerate(valloader)):
            with torch.no_grad():
                images_val = Variable(imgs_val.cuda())
                shade_labels_val = Variable(shades_val.cuda())

                # Val SE Network
                se_out_val = model(images_val)
                # ne_out_val = F.interpolate(ne_out_val, size=(
                #     256, 256), mode='bilinear', align_corners=True)
                # ne_pred_val = sigmoid(ne_out_val)
                se_pred_val = htan(se_out_val)

                ne_l1 = loss_fn(se_pred_val, shade_labels_val)
                ne_mse = MSE(se_pred_val, shade_labels_val)

                # Val SE Loss
                val_l1 += float(ne_l1.cpu())
                val_mse += float(ne_mse.cpu())

        len_valset = len(valloader)
        val_l1 = val_l1/len_valset
        val_mse = val_mse/len_valset
        val_losses = [val_l1, val_mse]
        print(f'SE L1 loss: {val_losses[0]} SE MSE: {val_losses[1]}')

        if tboard:
            writer.add_scalars('L1', {'train': loss, 'val': val_l1}, epoch)
            writer.add_scalars(
                'MSE', {'train': train_mse, 'val': val_mse}, epoch)

        # Reduce learning rate
        schedule.step(val_l1)

        if val_l1 < best_val_mse:
            best_val_mse = val_l1
            state = {'epoch': epoch + 1,
                     'model_state': model.state_dict(),
                     'optimizer_state': optimizer.state_dict()}
            torch.save(
                state, f'./checkpoints-se/unetnc_{epoch}_ne_{val_l1}_{loss}_best_model.pkl')

        if (epoch + 1) % 10 == 0:
            state = {'epoch': epoch + 1,
                     'model_state': model.state_dict(),
                     'optimizer_state': optimizer.state_dict()}
            torch.save(
                state, f'./checkpoints-se/unetnc_{epoch}_ne_auto_saving_every_ten_epochs_with_{val_l1}_{loss}_loss.pkl')


def test(img_path, model_path, show=False):
    imgorg = cv2.imread(img_path)
    imgorg = cv2.cvtColor(imgorg, cv2.COLOR_BGR2RGB)
    w, h = imgorg.shape[0], imgorg.shape[1]
    img = cv2.resize(imgorg, (256, 256))
    img = img[:, :, ::-1]
    img = img.astype(float) / 255
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, 0)
    img = torch.from_numpy(img).float()

    htan = nn.Hardtanh(0.0, 1.0)
    model = get_model('unetnc', n_classes=3, in_channels=3)
    state = convert_state_dict(torch.load(model_path)['model_state'])

    model.load_state_dict(state)
    model.eval()
    model.cuda()
    images = Variable(img.cuda())

    with torch.no_grad():
        output = model(images)
        pred = htan(output)

    pred = pred.cpu().detach().numpy()[0]

    if show:
        pred = pred.transpose((1, 2, 0))
        pred = cv2.resize(pred, (h, w), interpolation=cv2.INTER_NEAREST)
        norm = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)
        shade = cv2.cvtColor(pred, cv2.COLOR_BGR2GRAY)
        print(pred)
        _, axis = plt.subplots(1, 3)
        axis[0].imshow(imgorg)
        axis[1].imshow(norm)
        axis[2].imshow(shade, cmap='gray')
        plt.show()


if __name__ == "__main__":
    test('C:/Users/yuttapichai.lam/dev-environment/DewarpNet-master/eval/inp/3_4.jpg',
         './checkpoints-ne/unetnc_39_ne_0.028236698064380866_0.02608849649167208_best_model.pkl',
         show=True)
