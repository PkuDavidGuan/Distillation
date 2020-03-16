import torch
import math
import torch.nn as nn
from framework.utils.weight_init import *


def build_feature_connector(s_channel, t_channel):
    C = [nn.Conv2d(s_channel, t_channel, kernel_size=1, stride=1, padding=0, bias=False),
         nn.BatchNorm2d(t_channel)]

    for m in C:
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    return nn.Sequential(*C)


class MarginReLU(nn.Module):
    def __init__(self, cfg):
        super(MarginReLU, self).__init__()
        size_s = cfg['size_s']
        size_t = cfg['size_t']
        if cfg['margin_t'] is None:
            self.margin_t = None
        else:
            self.margin_t = [m.unsqueeze(1).unsqueeze(2).unsqueeze(0).cuda().detach() for m in cfg['margin_t']]
        assert len(size_s) == len(size_t)

        self.Connectors = nn.ModuleList([build_feature_connector(s[0], t[0]) for s, t in zip(size_s, size_t)])

    def forward(self, mat_s, mat_t):
        ret_s = []
        ret_t = []
        for i in range(len(mat_s)):
            ret_s.append(self.Connectors[i](mat_s[i]))

        # len(mat_s) may not equal to len(mat_t) in the AutoML method.
        for i in range(len(mat_t)):
            if self.margin_t is None:
                ret_t.append([mat_t[i], -1])
            else:
                ret_t.append([mat_t[i], self.margin_t[i]])

        return ret_s, ret_t
