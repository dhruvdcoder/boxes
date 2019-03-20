import torch
from torch import Tensor
from torch.nn import Module, Parameter
import torch.nn.functional as F
from .box_operations import *


class WeightedSum(Module):
    def __init__(self, num_models: int) -> None:
        super().__init__()
        self.weights = Parameter(torch.rand(num_models))

    def forward(self, box_vols: Tensor) -> Tensor:
        return (F.softmax(self.weights, dim=0).unsqueeze(0) @ box_vols).squeeze()

class BoxModel(Module):
    def __init__(self, num_models: int, num_boxes: int, dim: int, box_param = UnitBoxes, vol_func = clamp_volume,):
        super().__init__()
        self.boxes = box_param(num_models, num_boxes, dim)
        self.vol = vol_func
        self.int = intersection
        self.scale_func = lambda: self.vol(self.boxes.calc_dim_scales())
        self.weights = WeightedSum(num_models)

    def forward(self, ids):
        boxes = self.boxes(ids)
        z = boxes[:,:,:,0]
        Z = boxes[:,:,:,1]
        A_z = z[:,:,0]
        A_Z = Z[:,:,0]
        int_z, int_Z = self.int(z,Z)
        vol_int = self.vol(int_Z - int_z)
        vol_A = self.vol(A_Z - A_z)
        all_boxes = self.boxes()
        all_z = all_boxes[:,:,0]
        all_Z = all_boxes[:,:,1]
        unary_vol = self.vol(all_Z - all_z)
        return {
            'P(A,B)': self.weights(vol_int / self.scale_func()),
            'P(A)': self.weights(vol_A / self.scale_func()),
            'P(B|A)': self.weights(vol_int / vol_A),
            'z,Z': (z,Z),
            'boxes': boxes,
            'unary_vol': self.weights(unary_vol / self.scale_func()),
            'all_boxes': all_boxes,
            'all_z,all_Z': (all_z,all_Z),
            'boxes_param': self.boxes,
        }
