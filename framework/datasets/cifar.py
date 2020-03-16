import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets


def CIFAR_T(training):
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    if training:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                                              (4, 4, 4, 4), mode='reflect').squeeze()),
            transforms.ToPILImage(),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
    else:
        return  transforms.Compose([
        transforms.ToTensor(),
        normalize
        ])


def CIFAR(cfg):
    name = cfg['name']
    transform = cfg['transform']
    if cfg['training']:
        cifar = datasets.__dict__[name.upper()]('/data/datasets/cifar', train=True, download=True,
                                            transform=transform)
    else:
        cifar = datasets.__dict__[name.upper()]('/data/datasets/cifar', train=False, download=True,
                                                transform=transform)
    cifar.num_classes = int(name[5:])
    cifar.img_size = (32, 32)
    return cifar