import torch
from torch.nn import Module, Parameter


class DeltaBoxes(Module):

    def __init__(self, num_models: int, num_boxes: int, dim: int, **kwargs):
        super().__init__()
        self.z = Parameter(torch.rand(num_models, num_boxes, dim))
        self.logdelta = Parameter(torch.rand(num_models, num_boxes, dim))

    def forward(self, ids=slice(None, None, None), scaled = False, **kwargs):
        z = self.min(ids, scaled, **kwargs)
        Z = self.max(ids, scaled, **kwargs)
        return torch.stack((z,Z), dim=-2)

    def min(self, ids=slice(None, None, None), scaled = False, **kwargs):
        z = self.z[:, ids]
        if scaled:
            min_z = self.calc_min_z()
            dim_scales = self.calc_dim_scales()
            return (z - min_z) / dim_scales
        else:
            return z

    def max(self, ids=slice(None, None, None), scaled = False, **kwargs):
        Z = self.z[:, ids] + torch.exp(self.logdelta[:, ids])
        if scaled:
            min_z = self.calc_min_z()
            dim_scales = self.calc_dim_scales()
            return (Z - min_z) / dim_scales
        else:
            return Z

    def calc_min_z(self):
        min_z, _ = torch.min(self.z, dim=1)
        return min_z

    def calc_dim_scales(self):
        max_Z, _ = torch.max(self.z + torch.exp(self.logdelta), dim=1)
        min_z = self.calc_min_z()
        dim_scales = max_Z - min_z
        return dim_scales

