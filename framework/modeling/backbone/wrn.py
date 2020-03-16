import math
from scipy.stats import norm
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, name, num_classes, dropRate=0.0):
        model_config = name.split('-')
        depth = int(model_config[1])
        widen_factor = int(model_config[2])
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        strides = [1, 1, 2, 2] # if modify the stride, you should also modify ERF in function `get_ERF`
        self.nChannels = nChannels
        self.strides = strides
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) // 6
        self.num_per_block = n
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=strides[0],
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, strides[1], dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, strides[2], dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, strides[3], dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def get_size(self, img_size):
        feature_sizes = []
        out_size_h = img_size[0]
        out_size_w = img_size[1]
        for i in range(len(self.nChannels)):
            out_size_h = out_size_h // self.strides[i]
            out_size_w = out_size_w // self.strides[i]
            feature_sizes.append((self.nChannels[i], out_size_h, out_size_w))
        return feature_sizes

    def get_ERF(self, img_size):
        '''
        effective receptive field
        :param img_size:
        :return:
        '''
        scale = max(img_size)

        erf_stem = 3
        erf_block1 = erf_stem + 4 * self.num_per_block
        erf_block2 = erf_block1 + 6 + 8 * (self.num_per_block - 1)
        erf_block3 = erf_block2 + 12 + 16 * (self.num_per_block - 1)

        erfs = [erf_stem, erf_block1, erf_block2, erf_block3]
        for i in range(len(erfs)):
            erfs[i] = min(scale, erfs[i])
        jumps = [1, 1, 2, 4]
        return jumps, erfs

    def get_margin(self):
        def get_margin_from_BN(bn):
            margin = []
            std = bn.weight.data
            mean = bn.bias.data
            for (s, m) in zip(std, mean):
                s = abs(s.item())
                m = m.item()
                if norm.cdf(-m / s) > 0.001:
                    margin.append(- s * math.exp(- (m / s) ** 2 / 2) / math.sqrt(2 * math.pi) / norm.cdf(-m / s) + m)
                else:
                    margin.append(-3 * s)

            return torch.FloatTensor(margin).to(std.device)

        m0 = get_margin_from_BN(self.block1.layer[0].bn1)
        m1 = get_margin_from_BN(self.block2.layer[0].bn1)
        m2 = get_margin_from_BN(self.block3.layer[0].bn1)
        m3 = get_margin_from_BN(self.bn1)
        return [m0, m1, m2, m3]


    def forward(self, x, preReLU=False):
        g0 = self.conv1(x)
        g1 = self.block1(g0)
        g2 = self.block2(g1)
        g3 = self.block3(g2)
        out = self.relu(self.bn1(g3))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels[3])
        if preReLU:
            g0 = self.block1.layer[0].bn1(g0)
            g1 = self.block2.layer[0].bn1(g1)
            g2 = self.block3.layer[0].bn1(g2)
            g3 = self.bn1(g3)
        return self.fc(out),[g0,g1,g2,g3]
