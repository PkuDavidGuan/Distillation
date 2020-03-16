"""shufflenetv2 in pytorch
[1] Ningning Ma, Xiangyu Zhang, Hai-Tao Zheng, Jian Sun
    ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design
    https://arxiv.org/abs/1807.11164
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def channel_split(x, split):
    """split a tensor into two pieces along channel dimension
    Args:
        x: input tensor
        split:(int) channel size for each pieces
    """
    assert x.size(1) == split * 2
    return torch.split(x, split, dim=1)


def channel_shuffle(x, groups):
    """channel shuffle operation
    Args:
        x: input tensor
        groups: input branch number
    """

    batch_size, channels, height, width = x.size()
    channels_per_group = int(channels / groups)

    x = x.view(batch_size, groups, channels_per_group, height, width)
    x = x.transpose(1, 2).contiguous()
    x = x.view(batch_size, -1, height, width)

    return x


class ShuffleUnit(nn.Module):

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()

        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

        if stride != 1 or in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.ReLU(inplace=False),
                nn.Conv2d(in_channels, in_channels, 1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=False),
                nn.Conv2d(in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, int(out_channels / 2), 1),
                nn.BatchNorm2d(int(out_channels / 2))

            )

            self.shortcut = nn.Sequential(
                nn.ReLU(inplace=False),
                nn.Conv2d(in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, int(out_channels / 2), 1),
                nn.BatchNorm2d(int(out_channels / 2))

            )
        else:
            self.shortcut = nn.Sequential(nn.ReLU(inplace=False))

            in_channels = int(in_channels / 2)
            self.residual = nn.Sequential(
                nn.ReLU(inplace=False),
                nn.Conv2d(in_channels, in_channels, 1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=False),
                nn.Conv2d(in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, in_channels, 1),
                nn.BatchNorm2d(in_channels)

            )

    def forward(self, x):

        if self.stride == 1 and self.out_channels == self.in_channels:
            shortcut, residual = channel_split(x, int(self.in_channels / 2))
        else:
            shortcut = x
            residual = x

        shortcut = self.shortcut(shortcut)
        residual = self.residual(residual)
        x = torch.cat([shortcut, residual], dim=1)
        x = channel_shuffle(x, 2)

        return x


# stage=5
class ShuffleNetV2(nn.Module):

    def __init__(self, name, num_classes=100):
        model_cfg = name.split('-')
        ratio = float(model_cfg[1])
        super().__init__()
        if ratio == 0.5:
            out_channels = [48, 96, 192, 1024]
        elif ratio == 1:
            out_channels = [116, 232, 464, 1024]
        elif ratio == 1.5:
            out_channels = [176, 352, 704, 1024]
        elif ratio == 2:
            out_channels = [244, 488, 976, 2048]
        else:
            ValueError('unsupported ratio number')

        self.pre = nn.Sequential(
            nn.Conv2d(3, 24, 3, padding=1),
            nn.BatchNorm2d(24)
        )
        self.nChannels = [24] + out_channels
        self.prime_idx = [0, 4, 12, 16, 17]
        self.stage2 = self._make_stage(24, out_channels[0], 3)
        self.stage3 = self._make_stage(out_channels[0], out_channels[1], 7)
        self.stage4 = self._make_stage(out_channels[1], out_channels[2], 3)
        self.conv5 = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(out_channels[2], out_channels[3], 1),
            nn.BatchNorm2d(out_channels[3]),
            # nn.ReLU(inplace=True)
        )
        self.relu = nn.ReLU(inplace=False)
        self.fc = nn.Linear(out_channels[3], num_classes)

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x, preReLU=True, prime=True):

        g = [self.pre(x)]
        for i in range(len(self.stage2)):
            g.append(self.stage2[i](g[-1]))
        # x = self.stage2(x)
        for i in range(len(self.stage3)):
            g.append(self.stage3[i](g[-1]))
        for i in range(len(self.stage4)):
            g.append(self.stage4[i](g[-1]))
        # x = self.stage3(x)
        # x = self.stage4(x)
        g.append(self.conv5(g[-1]))
        out = self.relu(g[-1])
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        if prime:
            prime_g = []
            for idx in self.prime_idx:
                prime_g.append(g[idx])
            g = prime_g

        return out, g

    def partly_forward(self, x, stage):
        nn_layers = [self.pre, self.stage2, self.stage3, self.stage4, self.conv5]
        for i in range(stage + 1, len(nn_layers)):
            x = nn_layers[i](x)
        out = self.relu(x)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

    def _make_stage(self, in_channels, out_channels, repeat):
        layers = []
        layers.append(ShuffleUnit(in_channels, out_channels, 2))

        while repeat:
            layers.append(ShuffleUnit(out_channels, out_channels, 1))
            repeat -= 1

        return nn.Sequential(*layers)

    def get_prime_idx(self):
        return self.prime_idx

    def get_size(self, img_size):
        feature_sizes = []
        out_size_h = img_size[0]
        out_size_w = img_size[1]
        for i in range(len(self.nChannels)):
            feature_sizes.append((self.nChannels[i], out_size_h, out_size_w))
            out_size_h = out_size_h // 2
            out_size_w = out_size_w // 2
        return feature_sizes

    def get_margin(self):
        return None

def shufflenetv2():
    return ShuffleNetV2()
