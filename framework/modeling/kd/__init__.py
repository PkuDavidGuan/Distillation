from .identity import Identity
from .margin_ReLU import MarginReLU

__factory = {
    'margin_ReLU': MarginReLU,
    'elu': MarginReLU,
    'swish': MarginReLU,
}


def names():
    return sorted(__factory.keys())


def build_kd_model(cfg):
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
    if name not in __factory:
        return Identity(cfg)
    return __factory[name](cfg)
