import torch
from torch import Tensor # for type annotations
import torch.nn.functional as F
from typing import Tuple

def intersection(z:Tensor, Z:Tensor) -> Tuple[Tensor, Tensor]:
    '''
    :param z: Tensor(model, pair, [A/B], dim)
    :param Z: Tensor(model, pair, [A/B], dim)
    :return: Tuple(Tensor(model, pair, dim), Tensor(model,pair,dim)) of min/max of box for A intersect B
    '''
    z, _ = torch.max(z, dim=2)
    Z, _ = torch.min(Z, dim=2)
    return z, Z

def clamp_volume(sidelengths:Tensor) -> Tensor:
    """
    :param sidelengths: Tensor(model, box, dim)
    :return: Tensor(model, box) of volumes
    """
    return torch.prod((sidelengths).clamp(0), dim=2)

def soft_volume(sidelengths:Tensor) -> Tensor:
    """
    :param sidelengths: Tensor(model, box, dim)
    :return: Tensor(model, box) of volumes
    """
    return torch.prod(F.softplus(sidelengths), dim=2)

