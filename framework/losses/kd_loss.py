import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class KDLoss(nn.Module):
    def __init__(self, cfg):
        super(KDLoss, self).__init__()
        self.alpha = cfg["args"].alpha[0]
        self.temperature = cfg["args"].temperature
        self.reduction = 'mean'

    def step(self):
        return

    def forward(self, output_s, output_t, target):
        ce = F.cross_entropy(output_s, target, reduction='none')
        kd = -F.softmax(output_t / self.temperature) * F.log_softmax(output_s / self.temperature)
        kd = kd.sum(1)
        loss = self.alpha * ce + (1 - self.alpha) * (self.temperature ** 2) * kd
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduciton == 'sum':
            return loss.sum()
        else:
            return loss
