#init for different models
from .wrn import WideResNet
from .shufflenetV2 import ShuffleNetV2

__backbones = {
    'wrn':WideResNet,
    'shufflenetV2':ShuffleNetV2,
}


def build_model(cfg):
    args = cfg['args']
    model_cfg = cfg['name'].split('-')
    name = model_cfg[0]
    if name not in __backbones:
        raise NotImplementedError('The model {} is not implemented!'.format(model_cfg))
    return __backbones[name](cfg['name'], cfg['num_classes'])
