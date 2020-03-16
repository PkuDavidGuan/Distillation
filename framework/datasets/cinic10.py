import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets


def CINIC10_T(training):
    normalize = transforms.Normalize(mean=[0.47889522, 0.47227842, 0.43047404],
                                     std=[0.24205776, 0.23828046, 0.25874835])
    if training:
        return transforms.Compose([
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])


def CINIC10(cfg):
    cinic = torchvision.datasets.ImageFolder(cfg['root'], transform=cfg['transform'])
    cinic.num_classes = 10
    cinic.img_size = (32, 32)
    return cinic
