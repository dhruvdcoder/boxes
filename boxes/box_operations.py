import torch
from torch import Tensor # for type annotations
import torch.nn.functional as F
from typing import *
from .unit_boxes import UnitBoxes


def intersection(z:Tensor, Z:Tensor) -> Tuple[Tensor, Tensor]:
    """
    :param z: Tensor(model, pair, [A/B], dim)
    :param Z: Tensor(model, pair, [A/B], dim)
    :return: Tuple(Tensor(model, pair, dim), Tensor(model,pair,dim)) of min/max of box for A intersect B
    """
    z, _ = torch.max(z, dim=2)
    Z, _ = torch.min(Z, dim=2)
    return z, Z


def clamp_volume(sidelengths:Tensor) -> Tensor:
    """
    :param sidelengths: Tensor(model, box, dim)
    :return: Tensor(model, box) of volumes
    """
    return torch.prod((sidelengths).clamp(0), dim=-1)


def soft_volume(sidelengths:Tensor) -> Tensor:
    """
    :param sidelengths: Tensor(model, box, dim)
    :return: Tensor(model, box) of volumes
    """
    return torch.prod(F.softplus(sidelengths), dim=-1)


def detect_small_boxes(boxes: Tensor, vol_func: Callable = clamp_volume, min_vol: float = 1e-20) -> Tensor:
    """
    Returns the indices of boxes with volume smaller than eps.

    :param boxes: box parametrization as Tensor(model, box, z/Z, dim)
    :param vol_func: function taking in side lengths and returning volumes
    :param min_vol: minimum volume of boxes
    :return: masked tensor which selects boxes whose side lengths are less than min_vol
    """
    return vol_func(boxes[:,:,1] - boxes[:,:,0]) < min_vol


def replace_Z_by_cube(boxes: Tensor, indices: Tensor, cube_vol: float = 1e-20) -> Tensor:
    """
    Returns a new Z parameter for boxes for which those boxes[indices] are now replaced by cubes of size cube_vol

    :param boxes: box parametrization as Tensor(model, box, z/Z, dim)
    :param indices: box indices to replace by a cube
    :param cube_vol: volume of cube
    :return: tensor representing the Z parameter
    """
    return boxes[:, :, 0][indices] + cube_vol ** (1 / boxes.shape[-1])



def replace_Z_by_cube_(boxes: Tensor, indices: Tensor, cube_vol: float = 1e-20) -> Tensor:
    """
    Replaces the boxes indexed by `indices` by a cube of volume `min_vol` with the same z coordinate

    :param boxes: box parametrization as Tensor(model, box, z/Z, dim)
    :param indices: box indices to replace by a cube
    :param cube_vol: volume of cube
    :return: tensor representing the box parametrization with those boxes
    """
    boxes[:, :, 1][indices] = replace_Z_by_cube(boxes, indices, cube_vol)


def disjoint_boxes_mask(boxes: Tensor) -> Tensor:
    A = boxes[:,:,0]
    B = boxes[:,:,1]
    return ((B[:,:,1] <= A[:,:,0]) | (B[:,:,0] >= A[:,:,1])).any(dim=-1)


def overlapping_boxes_mask(boxes: Tensor) -> Tensor:
    return disjoint_boxes_mask(boxes) ^ 1


def containing_boxes_mask(boxes: Tensor) -> Tensor:
    """
    Returns a mask for when B contains A
    :param boxes:
    :return:
    """
    A = boxes[:,:,0]
    B = boxes[:,:,1]
    return ((B[:,:,1] >= A[:,:,1]) & (B[:,:,0] <= A[:,:,0])).all(dim=-1)


def needing_pull_mask(boxes: Tensor, target_prob_B_given_A: Tensor) -> Tensor:
    return (target_prob_B_given_A != 0) & disjoint_boxes_mask(boxes)


def needing_push_mask(boxes: Tensor, target_prob_B_given_A: Tensor) -> Tensor:
    return (target_prob_B_given_A != 1) & containing_boxes_mask(boxes)

