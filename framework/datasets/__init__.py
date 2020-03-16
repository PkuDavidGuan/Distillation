from .cifar import CIFAR, CIFAR_T
from .imagenet import ImageNet,ImageNet_T
from .cinic10 import CINIC10, CINIC10_T

__dataset = {
    'cifar10': CIFAR,
    'cifar100': CIFAR,
    'imagenet':ImageNet,
    'cinic10':CINIC10,
}

__transform = {
    'cifar10': CIFAR_T,
    'cifar100': CIFAR_T,
    'imagenet':ImageNet_T,
    'cinic10':CINIC10_T,
}


def build_dataset(cfg):
    """
    Create a dataset instance.
    Parameters
    ----------
    name : str
      The dataset name. Can be one of __factory
    root : str
      The path to the dataset directory.
    """
    name = cfg["name"]
    if name not in __dataset:
        raise NotImplementedError('There is no dataset named as {}.'.format(name))
    return __dataset[name](cfg)


def build_transform(name, training):
    if name not in __transform:
        raise NotImplementedError('Dataset {} has no transform method.'.format(name))
    return __transform[name](training)
