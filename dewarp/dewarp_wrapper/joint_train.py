import torch
import torch.nn as nn
import torch.nn.functional as F


from models import get_model
import recon_lossc
import grad_loss
import joint_loss

def train(n_epoch=50, resume=False, wc_path='', bm_path=''):
    wc_model_name = 'unetnc'
    bm_model_name = 'dnetccnl'
    
    wc_n_classes = 3
    bm_n_classes = 2
    
    wc_img_size = (256, 256)
    bm_img_size = (128, 128)
    
    #Last layer activation
    htan = nn.Hardtanh(0, 1.0)
    
    #Load models
    wc_model = get_model(wc_model_name, n_classes=wc_n_classes, in_channels=3)
    wc_model.cuda()
    bm_model = get_model(bm_model_name, n_classes=bm_n_classes, in_channels=3)
    bm_model.cuda()
    
    #Setup optimizer and learning rate reduction
    optimizer = torch.optim.Adam([{'params': wc_model.parameters()},
                                 {'params': bm_model.parameters()}], lr=1e-4, weight_decay=5e-4, amsgrad=True)
    schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4, verbose=True)
    
    #Setup losses
    MSE = nn.MSELoss()
    loss_fn = nn.L1Loss()
    reconst_loss = recon_lossc.Unwarploss()
    g_loss = grad_loss.Gradloss(window_size=5, padding=2)
    jloss_fn = joint_loss.JointLoss()
    
    epoch_start = 0
    
    if resume:
        wc_chkpnt = torch.load(wc_path)
        wc_model.load_state_dict(wc_chkpnt['model_state'])
        bm_chkpnt = torch.load(bm_path)
        bm_model.load_state_dict(bm_chkpnt['model_state'])
        epoch_start = bm_chkpnt['epoch']
    
    best_valwc_mse = 9999999.0
    best_valbm_mse = 9999999.0
    
    for epoch in range(epoch_start, n_epoch):
        #Loss implementation
        
        #Start training
        wc_model.train()
        bm_model.train()
        
        raise NotImplementedError