import torch
import torch.nn as nn
import torch.nn.functional as F


class UnitBoxes(nn.Module):

    def __init__(self, num_models:int, num_boxes:int, dim:int):
        '''
        Uniformly random distribution of coordinates, ensuring
            Z_j > z_j for all j.
        '''
        super().__init__()
        box_mins = torch.rand(num_models, num_boxes, dim)
        box_maxs = box_mins + torch.rand(num_models, num_boxes, dim) * (1 - box_mins)
        self.boxes = Parameter(torch.stack([box_mins, box_maxs], dim=2))


    def min_max(self, ids=None):
        return self.boxes[:,ids]

    def min(self, ids=None):
        if ids is None:
            return self.boxes[:,:,0]
        else:
            return self.boxes[:,ids,0]

    def max(self, ids=None):
        if ids is None:
            return self.boxes[:,:,1]
        else:
            return self.boxes[:,:,0]
