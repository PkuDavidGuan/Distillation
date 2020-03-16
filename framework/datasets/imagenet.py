from torch.utils.data import Dataset
import numpy as np
import sys
import boto3
import torchvision.transforms as transforms
import torch
import cv2
from matplotlib import pyplot as plt
import PIL
from PIL import Image
import io


def ImageNet_T(training):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if training:
        transform = transforms.Compose(
            [transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
    else:
        transform = transforms.Compose(
            [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), normalize])
    return transform


class ImageNet(Dataset):
    def __init__(self, cfg):
        super(ImageNet, self).__init__()
        counter = 0
        self.root = cfg['root']
        self.transform = cfg['transform']
        self.num_classes = 1000
        self.img_size = (224, 224)
        self.samples = []

        self._data_loading()

    def _data_loading(self):
        '''
        load all the pairs (filename_of_a_image:str, label:int) into self.samples.
        :return:
        '''
        raise NotImplementedError

    def __getitem__(self, index):
        # assert index <= len(self), 'index range error'
        fname, label = self.samples[index]
        imgdata = Image.open(fname)
        # print(imgdata.size)
        img = self.transform(imgdata)
        return img, label

    def __len__(self):
        return len(self.samples)
