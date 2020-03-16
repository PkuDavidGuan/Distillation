import torch
import torch.nn as nn
import torch.nn.functional as F
import datetime


class SPLoss(nn.Module):
    def __init__(self, cfg):
        super(SPLoss, self).__init__()
        self.gamma = 3000

    def forward(self, output, target, mat_s, mat_t):
        B = output.size(0)

        ce = F.cross_entropy(output, target)
        sp = 0
        for i in range(len(mat_s)):
            g_s = self.get_g(mat_s[i])
            g_t = self.get_g(mat_t[i])
            sp += (torch.norm(g_t - g_s, p='fro')) ** 2
        sp *= 1 / (B ** 2)
        return ce + self.gamma * sp

    def get_g(self, mat):
        B = mat.size(0)
        mat = mat.view(B, -1)
        mat_t = torch.transpose(mat, 0, 1)
        g = torch.matmul(mat, mat_t)
        g_norm = torch.norm(g, dim=1).unsqueeze(1)
        return g / g_norm



