import torch
import torch.nn as nn
import torch.nn.functional as F


class MarginReLULoss(nn.Module):
    def __init__(self, cfg):
        super(MarginReLULoss, self).__init__()
        self.alpha = 1e-3

    def forward(self, output, target, mat_s, mat_t):
        B = output.size(0)

        ce = F.cross_entropy(output, target)

        feat_num = len(mat_s)
        distill_loss = 0
        for i in range(feat_num):
            distill_loss += self.distillation_loss(mat_s[i], mat_t[i][0], mat_t[i][1]) / 2 ** (feat_num - i - 1)
        distill_loss /= B
        return ce + self.alpha * distill_loss

    def distillation_loss(self, source, target, margin):
        loss = ((source - margin) ** 2 * ((source > margin) & (target <= margin)).float() +
                (source - target) ** 2 * ((source > target) & (target > margin) & (target <= 0)).float() +
                (source - target) ** 2 * (target > 0).float())
        return torch.abs(loss).sum()