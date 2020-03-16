import torch
import torch.nn as nn
import torch.nn.functional as F


class ATLoss(nn.Module):
    def __init__(self, cfg):
        super(ATLoss, self).__init__()
        self.beta = 60

    def forward(self, output, target, mat_s, mat_t):
        B = output.size(0)

        ce = F.cross_entropy(output, target)

        num_kd = len(mat_s)

        q_s = []
        q_t = []
        for i in range(num_kd):
            q_s.append(torch.sum(mat_s[i], dim=1))
            q_t.append(torch.sum(mat_t[i], dim=1))

        at = 0
        for i in range(num_kd):
            at += F.mse_loss(self.normalize(q_s[i]), self.normalize(q_t[i]))

        return ce + (self.beta / 2) * at

    def normalize(self, m, p=2):
        batch_size = m.size(0)
        norm = torch.norm(m.view(batch_size, -1), p=p, dim=1).view(batch_size, 1, 1, 1)
        return m / norm