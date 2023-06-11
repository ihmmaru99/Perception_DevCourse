import torch
import torch.nn as nn
import sys

class MNISTloss(nn.Module):
    def __init__(self, device=torch.device('cpu')):
        super(MNISTloss, self).__init__()
        self.loss = nn.CrossEntropyLoss().to(device)

    def forward(self, out, gt):
        loss_val = self.loss(out, gt)
        return loss_val
    
def get_criterion(crit="mnist", device=torch.device('cpu')):
    if crit is 'mnist':
        return MNISTloss(device=device)
    else:
        print("unknown criterion")
        sys.exit()