import torch
import torch.nn as nn


class JointLoss(torch.nn.Module):
    def __init__(self):
        super(JointLoss, self).__init__()

    def forward(self, pred, label):
        (batch_size, channel, h, w) = pred.size()
        batch_mask = (label != 0).float()
        fg_pred = pred * batch_mask
        bg_pred = pred * (1 - batch_mask)
        bg_label = label * (1 - batch_mask)

        # Get the BCE Loss for the foreground
        fg_loss_fn = nn.Loss()

        # Get the hinge loss for background
        bg_loss = torch.mean(
            torch.max(torch.max(torch.max(bg_pred, bg_label), dim=-1)[0], dim=-1)[0])
        fg_loss = fg_loss_fn(fg_pred, label)

        return bg_loss, fg_loss
