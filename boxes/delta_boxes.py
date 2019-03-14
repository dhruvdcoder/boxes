import torch
import torch.nn as nn
import torch.nn.functional as F


class DeltaBoxes(nn.Module):

    def __init__(self, num_models: int, num_boxes: int, dim: int):
        super().__init__()
        self.z = nn.Parameter(torch.rand(num_models, num_boxes, dim))
        self.logdelta = nn.Parameter(torch.rand(num_models, num_boxes, dim))

    def min(self, ids=None, scaled=True):
        if ids is None:
            return self.z
        else:
            return self.z[:,ids]

    def max(self, ids=None, scaled=True):
        if ids is None:
            return self.z + torch.exp(self.logdelta)
        else:
            return self.z[:,ids] + torch.exp(self.logdelta[:,ids])

    def calc_dim_scales(z, logdelta):
        max_Z, _ = torch.max(z + torch.exp(logdelta), dim=1)
        min_z, _ = torch.min(z, dim=1)
        self.dim_scales = max_Z - min_z
        return self.dim_scales
