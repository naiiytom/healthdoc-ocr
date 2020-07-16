import torch
from torch.autograd import Variable
from torch.cuda import device_count
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from tqdm import tqdm

from models import get_model
from loaders import get_loader
import recon_lossc
import grad_loss


def train(n_epoch=50, batch_size=32, resume=False, wc_path='', bm_path=''):
    wc_model_name = 'unetnc'
    bm_model_name = 'dnetccnl'

    # Setup dataloader
    data_path = 'D:/doc3d-dataset'
    data_loader = get_loader('doc3djoint')
    t_loader = data_loader(data_path, is_transform=True,
                           img_size=(256, 256), bm_size=(128, 128))
    v_loader = data_loader(data_path, split='val',
                           is_transform=True, img_size=(256, 256), bm_size=(128, 128))

    trainloader = data.DataLoader(
        t_loader, batch_size=batch_size, num_workers=8, shuffle=True)
    valloader = data.DataLoader(v_loader, batch_size=batch_size, num_workers=8)

    # Last layer activation
    htan = nn.Hardtanh(0, 1.0)

    # Load models
    print('Loading')
    wc_model = get_model(wc_model_name, n_classes=3, in_channels=3)
    wc_model = torch.nn.DataParallel(
        wc_model, device_ids=range(torch.cuda.device_count()))
    wc_model.cuda()
    bm_model = get_model(bm_model_name, n_classes=2, in_channels=3)
    bm_model = torch.nn.DataParallel(
        bm_model, device_ids=range(torch.cuda.device_count()))
    bm_model.cuda()

    # Setup optimizer and learning rate reduction
    print('Setting optimizer')
    optimizer = torch.optim.Adam([{'params': wc_model.parameters()},
                                  {'params': bm_model.parameters()}], lr=1e-4, weight_decay=5e-4, amsgrad=True)
    schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=4, verbose=True)

    # Setup losses
    MSE = nn.MSELoss()
    loss_fn = nn.L1Loss()
    reconst_loss = recon_lossc.Unwarploss()
    g_loss = grad_loss.Gradloss(window_size=5, padding=2)

    epoch_start = 0

    if resume:
        print('Resume from previous state')
        wc_chkpnt = torch.load(wc_path)
        wc_model.load_state_dict(wc_chkpnt['model_state'])
        bm_chkpnt = torch.load(bm_path)
        bm_model.load_state_dict(bm_chkpnt['model_state'])
        # optimizer.load_state_dict(
        #     [wc_chkpnt['optimizer_state'], bm_chkpnt['optimizer_state']])
        epoch_start = bm_chkpnt['epoch']

    best_valwc_mse = 9999999.0
    best_valbm_mse = 9999999.0
    print(f'Start from epoch {epoch_start} of {n_epoch}')
    print('Starting')
    for epoch in range(epoch_start, n_epoch):
        print(f'Epoch: {epoch}')
        # Loss initialization
        avg_loss = 0.0
        avg_wcloss = 0.0
        avgwcl1loss = 0.0
        avg_gloss = 0.0
        train_wcmse = 0.0
        avg_bmloss = 0.0
        avgbml1loss = 0.0
        avgrloss = 0.0
        avgssimloss = 0.0
        train_bmmse = 0.0

        # Start training
        wc_model.train()
        bm_model.train()

        print('Training')

        for i, (imgs, wcs, bms, recons) in enumerate(trainloader):
            images = Variable(imgs.cuda())
            wc_labels = Variable(wcs.cuda())
            bm_labels = Variable(bms.cuda())
            recon_labels = Variable(recons.cuda())

            optimizer.zero_grad()

            # Train WC network
            wc_out = wc_model(images)
            wc_out = F.interpolate(wc_out, size=(
                256, 256), mode='bilinear', align_corners=True)
            bm_inp = F.interpolate(wc_out, size=(
                128, 128), mode='bilinear', align_corners=True)
            bm_inp = htan(bm_inp)
            wc_pred = htan(wc_out)

            wc_l1loss = loss_fn(wc_pred, wc_labels)
            wc_gloss = g_loss(wc_pred, wc_labels)
            wc_mse = MSE(wc_pred, wc_labels)
            wc_loss = wc_l1loss + (0.2 * wc_gloss)

            # WC Loss
            avgwcl1loss += float(wc_l1loss)
            avg_gloss += float(wc_gloss)
            train_wcmse += float(wc_mse)
            avg_wcloss += float(wc_loss)

            # Train BM network
            bm_out = bm_model(bm_inp)
            bm_out = bm_out.transpose(1, 2).transpose(2, 3)

            bm_l1loss = loss_fn(bm_out, bm_labels)
            rloss, ssim, uworg, uwpred = reconst_loss(
                recon_labels, bm_out, bm_labels)
            bm_mse = MSE(bm_out, bm_labels)
            bm_loss = (10 * bm_l1loss) + (0.5 * rloss)

            # BM Loss
            avgbml1loss += float(bm_l1loss)
            avgrloss += float(rloss)
            avgssimloss += float(ssim)
            train_bmmse += float(bm_mse)
            avg_bmloss += float(bm_loss)

            # Step loss
            loss = (0.5 * wc_loss) + (0.5 * bm_loss)
            avg_loss += float(loss)

            if (i+1) % 10 == 0:
                print(
                    f'Epoch[{epoch}/{n_epoch}] Batch[{i+1}/{len(trainloader)}] Loss: {avg_loss/(i+1):.6f}')

            loss.backward()
            optimizer.step()

        len_trainset = len(trainloader)
        train_wcmse = train_wcmse/len_trainset
        train_bmmse = train_bmmse/len_trainset
        train_losses = [avgwcl1loss/len_trainset, train_wcmse, avg_gloss/len_trainset,
                        avgbml1loss/len_trainset, train_bmmse, avgrloss/len_trainset, avgssimloss/len_trainset]
        print(
            f'WC L1 loss: {train_losses[0]} WC MSE: {train_losses[1]} WC GLoss: {train_losses[2]}')
        print(
            f'BM L1 Loss: {train_losses[3]} BM MSE: {train_losses[4]} BM RLoss: {train_losses[5]} BM SSIM Loss: {train_losses[6]}')
        wc_model.eval()
        bm_model.eval()

        wc_val_l1 = 0.0
        wc_val_mse = 0.0
        wc_val_gloss = 0.0
        bm_val_l1 = 0.0
        bm_val_mse = 0.0
        bm_val_rloss = 0.0
        bm_val_ssim = 0.0

        print('Validating')

        for i_val, (imgs_val, wcs_val, bms_val, recons_val) in tqdm(enumerate(valloader)):
            with torch.no_grad():
                images_val = Variable(imgs_val.cuda())
                wc_labels_val = Variable(wcs_val.cuda())
                bm_labels_val = Variable(bms_val.cuda())
                recon_labels_val = Variable(recons_val.cuda())

                # Val WC Network
                wc_out_val = wc_model(images_val)
                wc_out_val = F.interpolate(wc_out_val, size=(
                    256, 256), mode='bilinear', align_corners=True)
                bm_inp_val = F.interpolate(wc_out_val, size=(
                    128, 128), mode='bilinear', align_corners=True)
                bm_inp_val = htan(bm_inp_val)
                wc_pred_val = htan(wc_out_val)

                wc_l1 = loss_fn(wc_pred_val, wc_labels_val)
                wc_gloss = g_loss(wc_pred_val, wc_labels_val)
                wc_mse = MSE(wc_pred_val, wc_labels_val)

                # Val BM network
                bm_out_val = bm_model(bm_inp_val)
                bm_out_val = bm_out_val.transpose(1, 2).transpose(2, 3)

                bm_l1 = loss_fn(bm_out_val, bm_labels_val)
                rloss, ssim, uworg, uwpred = reconst_loss(
                    recon_labels_val, bm_out_val, bm_labels_val)
                bm_mse = MSE(bm_out_val, bm_labels_val)

                # Val Loss
                wc_val_l1 += float(wc_l1.cpu())
                wc_val_gloss += float(wc_gloss.cpu())
                wc_val_mse += float(wc_mse.cpu())

                bm_val_l1 += float(bm_l1.cpu())
                bm_val_mse += float(bm_mse.cpu())
                bm_val_rloss += float(rloss.cpu())
                bm_val_ssim += float(ssim.cpu())

        len_valset = len(valloader)
        wc_val_mse = wc_val_mse/len_valset
        bm_val_mse = bm_val_mse/len_valset
        val_losses = [wc_val_l1/len_valset, wc_val_mse, wc_val_gloss/len_valset,
                      bm_val_l1/len_valset, bm_val_mse, bm_val_rloss/len_valset, bm_val_ssim/len_valset]
        print(
            f'WC L1 loss: {val_losses[0]} WC MSE: {val_losses[1]} WC GLoss: {val_losses[2]}')
        print(
            f'BM L1 Loss: {val_losses[3]} BM MSE: {val_losses[4]} BM RLoss: {val_losses[5]} BM SSIM Loss: {val_losses[6]}')
        # Reduce learning rate
        schedule.step(wc_val_mse)

        if wc_val_mse < best_valwc_mse:
            best_valwc_mse = wc_val_mse
            state = {'epoch': epoch,
                     'model_state': wc_model.state_dict()}
            torch.save(
                state, f'./checkpoints-wc/unetnc_{epoch}_wc_{wc_val_mse}_{train_wcmse}_best_model.pkl')

        if bm_val_mse < best_valbm_mse:
            best_valbm_mse = bm_val_mse
            state = {'epoch': epoch,
                     'model_state': bm_model.state_dict()}
            torch.save(
                state, f'./checkpoints-bm/dnetccnl_{epoch}_bm_{bm_val_mse}_{train_bmmse}_best_model.pkl')


if __name__ == "__main__":
    train(n_epoch=120, batch_size=64, resume=True,
          wc_path='C:/Users/yuttapichai.lam/dev-environment/pretrained-models/unetnc_doc3d.pkl',
          bm_path='C:/Users/yuttapichai.lam/dev-environment/pretrained-models/dnetccnl_doc3d.pkl')
