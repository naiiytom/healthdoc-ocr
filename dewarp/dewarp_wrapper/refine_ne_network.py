import torch
from torch.autograd import Variable
from torch.cuda import device_count
from torch.functional import norm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from tqdm import tqdm

from models import get_model
from loaders import get_loader


def train(n_epoch=50, batch_size=32, resume=False):
    model_name = 'unetnc'

    # Setup dataloader
    data_path = 'D:/doc3d-dataset'
    data_loader = get_loader('ne_refine')
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

    # Setup optimizer and learning rate reduction
    print('Setting optimizer')
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-4, weight_decay=5e-4, amsgrad=True)
    schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=4, verbose=True)

    # Set Activation function
    htan = nn.Hardtanh(0, 1.0)

    # Setup losses
    MSE = nn.MSELoss()
    loss_fn = nn.L1Loss()

    epoch_start = 0

    # if resume:
    #     print('Resume from previous state')
    #     wc_chkpnt = torch.load(wc_path)
    #     wc_model.load_state_dict(wc_chkpnt['model_state'])
    #     # optimizer.load_state_dict(
    #     #     [wc_chkpnt['optimizer_state'], bm_chkpnt['optimizer_state']])

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

        for i, (imgs, norms) in enumerate(trainloader):
            images = Variable(imgs.cuda())
            norm_labels = Variable(norms.cuda())

            optimizer.zero_grad()

            # Train NE network
            ne_out = model(images)
            ne_out = F.interpolate(ne_out, size=(
                256, 256), mode='bilinear', align_corners=True)
            ne_pred = htan(ne_out)

            l1_loss = loss_fn(ne_pred, norm_labels)
            mse = MSE(ne_pred, norm_labels)

            # NE Loss
            avg_l1loss += float(l1_loss)
            train_mse += float(mse)
            avg_loss += float(l1_loss)

            if (i+1) % 10 == 0:
                print(
                    f'Epoch[{epoch}/{n_epoch}] Batch[{i+1}/{len(trainloader)}] Loss: {avg_loss/(i+1):.6f}')

            l1_loss.backward()
            optimizer.step()

        len_trainset = len(trainloader)
        train_wcmse = train_mse/len_trainset
        train_losses = [avg_l1loss/len_trainset, train_wcmse]
        print(f'NE L1 loss: {train_losses[0]} NE MSE: {train_losses[1]}')

        model.eval()
        val_l1 = 0.0
        val_mse = 0.0

        print('Validating')

        for i_val, (imgs_val, norms_val) in tqdm(enumerate(valloader)):
            with torch.no_grad():
                images_val = Variable(imgs_val.cuda())
                norm_labels_val = Variable(norms_val.cuda())

                # Val NE Network
                ne_out_val = model(images_val)
                ne_out_val = F.interpolate(ne_out_val, size=(
                    256, 256), mode='bilinear', align_corners=True)
                ne_pred_val = htan(ne_out_val)

                ne_l1 = loss_fn(ne_pred_val, norm_labels_val)
                ne_mse = MSE(ne_pred_val, norm_labels_val)

                # Val NE Loss
                val_l1 += float(ne_l1.cpu())
                val_mse += float(ne_mse.cpu())

        len_valset = len(valloader)
        wc_val_mse = val_mse/len_valset
        val_losses = [val_l1/len_valset, wc_val_mse]
        print(f'WC L1 loss: {val_losses[0]} WC MSE: {val_losses[1]}')

        # Reduce learning rate
        schedule.step(val_mse)

        if val_mse < best_val_mse:
            best_val_mse = wc_val_mse
            state = {'epoch': epoch,
                     'model_state': model.state_dict()}
            torch.save(
                state, f'./checkpoints-ne/unetnc_{epoch}_ne_{val_mse}_{train_mse}_best_model.pkl')


def test():
    pass


if __name__ == "__main__":
    train(n_epoch=10, batch_size=2)
