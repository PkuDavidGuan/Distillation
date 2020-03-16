from .kd_loss import KDLoss
from .margin_ReLU_loss import MarginReLULoss
from .similarity_loss import SPLoss
from .act_loss import *
from .at_loss import ATLoss


__factory = {
    'kd': KDLoss,
    'sp': SPLoss,
    'margin_ReLU': MarginReLULoss,
    'elu': ELULoss,
    'swish': SwishLoss,
    'at': ATLoss,
}


def names():
  return sorted(__factory.keys())


def build_criterion(cfg):
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
    raise NotImplementedError('The method does not have its own loss calculation method.')
  return __factory[name](cfg)
