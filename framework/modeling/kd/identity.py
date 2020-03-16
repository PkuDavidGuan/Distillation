import torch
import torch.nn as nn


class Identity(nn.Module):
    def __init__(self, cfg):
        super(Identity, self).__init__()

    def forward(self, mat_s, mat_t):
        return mat_s, mat_t