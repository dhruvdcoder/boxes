import torch
from torch.nn import Module, Parameter


class UnitBoxes(Module):

    def __init__(self, num_models:int, num_boxes:int, dim:int, init_min_vol = 1e-2):
        '''
        Uniformly random distribution of coordinates, ensuring
            Z_j > z_j for all j.
        '''
        super().__init__()
        if init_min_vol == 0:
            '''
            Uniformly random distribution of coordinates, ensuring
                Z_j > z_j for all j.
            '''
            z = torch.rand(num_models, num_boxes, dim)
            Z = z + torch.rand(num_models, num_boxes, dim) * (1. - z)
        elif init_min_vol > 0:
            '''
            Uniformly random distribution of coordinates, ensuring that each box
            contains a cube of volume larger than init_min_vol.
            Useful in higher dimensions to deal with underflow.
            '''
            eps = init_min_vol ** (1 / dim)
            z = torch.rand(num_models, num_boxes, dim) * (1 - eps)
            Z = z + eps + torch.rand(num_models, num_boxes, dim) * (1 - (z + eps))
        else:
            raise ValueError(f'init_min_vol={init_min_vol} is an invalid option.')

        self.boxes = Parameter(torch.stack((z, Z), dim=2))

    def min(self, ids=slice(None, None, None), **kwargs):
        return self.boxes[:,ids,0]

    def max(self, ids=slice(None, None, None), **kwargs):
        return self.boxes[:,ids,1]
