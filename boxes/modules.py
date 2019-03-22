import torch
from torch import Tensor
from torch.nn import Module, Parameter
import torch.nn.functional as F
from .box_operations import *


################################################
# Box Embedding Layers
################################################

class BoxEmbedding(Module):
    """
    A base class for creating a Box Embedding parametrization.
    You should explicitly inherit from both Module and BoxEmbedding, i.e.

    ```
    class MyBoxEmbedding(Module, BoxEmbedding):
        def __init__(self, num_models:int, num_boxes:int, dim:int, **kwargs):
            super().__init__() # <- This will refer to Module.__init__()
            ...
    ```
    """

    def __init__(self, num_models:int, num_boxes:int, dim:int, vol_func:Callable, **kwargs):
        """
        Creates the Parameters used for the representation of boxes.

        :param num_models: Number of models
        :param num_boxes: Number of boxes
        :param dim: Dimension
        :param vol_func: Function which can take side-lengths -> volumes
        :param kwargs: Unused for now, but include this for future possible parameters.
        """
        # Remember to call:
        super().__init__()
        raise NotImplemented

    def forward(self, ids: Tensor, scaled = False, **kwargs) -> Tensor:
        """
        Returns a Tensor representing the pairs of boxes specified by ids.

        :param ids: A list or 1-dimensional Tensor
        :param scaled: If True, return the coordinates scaled to the [0,1]^d hypercube.
        :param kwargs: Unused for now, but include this for future possible parameters.
        :return: Tensor of shape (model, id, min/max, dim).
        """
        raise NotImplemented


    # def min(self, ids: Tensor = slice(None, None, None), scaled = False, **kwargs) -> Tensor: """
    #     Returns the min coordinates for the pairs of boxes specified by id_pairs.
    #
    #     :param ids: For single boxes, a 1-dimensional Tensor.
    #                 For pairs of boxes, a Tensor of shape (num_pairs, 2)
    #                 If unset, equivalent to [:], i.e. return all boxes.
    #     :param scaled: If True, return the coordinates scaled to the [0,1]^d hypercube.
    #     :param kwargs: Unused for now, but include this for future possible parameters.
    #     :return: Tensor of shape (model, *ids.shape, dim).
    #     """
    #     raise NotImplemented
    #
    #
    # def max(self, ids: Tensor = slice(None, None, None), scaled = False, **kwargs) -> Tensor:
    #     """
    #     Returns the max coordinates for the pairs of boxes specified by id_pairs.
    #
    #     :param ids: For single boxes, a 1-dimensional Tensor.
    #                 For pairs of boxes, a Tensor of shape (num_pairs, 2)
    #                 If unset, equivalent to [:], i.e. return all boxes.
    #     :param scaled: If True, return the coordinates scaled to the [0,1]^d hypercube.
    #     :param kwargs: Unused for now, but include this for future possible parameters.
    #     :return: Tensor of shape (model, pair, A/B, dim)
    #     """
    #     raise NotImplemented

    # def universe_min(self):
    #     """
    #     Returns the minimum coordinate for the "universe" in each model in each dimension.
    #         - For boxes which enforce a parametrization in [0,1] this would just be 0.
    #         - For boxes which allow a dynamic universe, this is typically the smallest
    #           min coordinate.
    #
    #     :return: Tensor (model, dim)
    #     """
    #     return torch.min(self.min(), dim=1)
    #
    # def universe_max(self):
    #     """
    #     Returns the maximum coordinate for each model in each dimension.
    #         - For boxes which enforce a parametrization in [0,1] this would just be 1.
    #         - For boxes which allow a dynamic universe, this is typically the smallest
    #           max coordinate.
    #
    #     :return: Tensor (model, dim)
    #     """
    #     return torch.max(self.max(), dim=1)

    def universe_box(self) -> Tensor:
        """
        For each model, returns the min/max coordinates of the smallest box which contains
        all other boxes.
            - For boxes which enforce a parametrization in [0,1] this would just be 1.
            - For boxes which allow a dynamic universe, this is typically the smallest
              max coordinate.

        :return: Tensor (model, min/max, dim)
        """
        return torch.stack((self.min_coord(), self.max_coord()), dim=1)


class UnitBoxes(Module, BoxEmbedding):
    """
    Parametrize boxes using the min coordinate and max coordinate,
    initialized to be in the unit hypercube.

    self.boxes[model, box, min/max, dim] \in [0,1]

    In this parametrization, the min and max coordinates are explicitly stored
    in separate dimensions (as shown above), which means that care must be
    taken to preserve max > min while training. (See MinBoxSize Callback.)
    """

    def __init__(self, num_models:int, num_boxes:int, dim:int, init_min_vol = 1e-2, **kwargs):
        super().__init__()
        if init_min_vol == 0:
            """
            Uniformly random distribution of coordinates, ensuring
                Z_j > z_j for all j.
            """
            z = torch.rand(num_models, num_boxes, dim)
            Z = z + torch.rand(num_models, num_boxes, dim) * (1. - z)
        elif init_min_vol > 0:
            """
            Uniformly random distribution of coordinates, ensuring that each box
            contains a cube of volume larger than init_min_vol.
            Useful in higher dimensions to deal with underflow, however in high
            dimensions a cube with small volume still has very large side-lengths.
            """
            eps = init_min_vol ** (1 / dim)
            z = torch.rand(num_models, num_boxes, dim) * (1 - eps)
            Z = z + eps + torch.rand(num_models, num_boxes, dim) * (1 - (z + eps))
        else:
            raise ValueError(f'init_min_vol={init_min_vol} is an invalid option.')

        self.boxes = Parameter(torch.stack((z, Z), dim=2))

    def forward(self, ids=slice(None, None, None), **kwargs):
        return self.boxes[:,ids]

    def universe_box(self):
        return torch.ones(self.boxes.shape[-1]).to(self.boxes.device)


class DeltaBoxes(Module, BoxEmbedding):
    """
    Parametrize boxes using the min coordinate and log of side-length.

    self.z[model, box, dim] \in \RR
    self.logdelta[model, box, dim] \in \RR

    This forces boxes to always have positive side-lengths.
    """

    def __init__(self, num_models: int, num_boxes: int, dim: int, **kwargs):
        super().__init__()
        self.z = Parameter(torch.rand(num_models, num_boxes, dim))
        self.logdelta = Parameter(torch.rand(num_models, num_boxes, dim))

    def forward(self, ids = slice(None, None, None), scaled = False, **kwargs):
        z = self.min(ids, scaled, **kwargs)
        Z = self.max(ids, scaled, **kwargs)
        return torch.stack((z,Z), dim=-2)

    def min(self, ids = slice(None, None, None), scaled = False, **kwargs):
        z = self.z[:, ids]
        if scaled:
            return (z - min_z) / dim_scales
        else:
            return z

    def max(self, ids = slice(None, None, None), scaled = False, **kwargs):
        Z = self.z[:, ids] + torch.exp(self.logdelta[:, ids])
        if scaled:
            min_z = self.calc_min_z()
            dim_scales = self.calc_dim_scales()
            return (Z - min_z) / dim_scales
        else:
            return Z


class WeightedSum(Module):
    def __init__(self, num_models: int) -> None:
        super().__init__()
        self.weights = Parameter(torch.rand(num_models))

    def forward(self, box_vols: Tensor) -> Tensor:
        return (F.softmax(self.weights, dim=0).unsqueeze(0) @ box_vols).squeeze()


class CondProbs(Module):
    def __init__(self, box_embedding: BoxEmbedding, vol_func = clamp_volume):
        super().__init__()
        self.box_embedding = box_embedding
        self.vol = vol_func

    def forward(self, ids):
        A = self.box_embedding(ids[0])
        B = self.box_embedding(ids[1])
        vol_A_int_B = self.vol_func(intersection(A, B))
        vol_A = self.vol_func(A)
        return {
            'P(B|A)': vol_A_int_B / vol_A,
            'A': A,
            'B': B,
        }


class UnaryProbs(Module):

    def __init__(self, box_embedding: BoxEmbedding, vol_func = clamp_volume):
        super().__init__()
        self.box_embedding = box_embedding
        self.vol = vol_func

    def forward(self):
        U = self.box_embedding()
        vol_U = self.vol_func(U)
        vol_universe = self.vol_func(self.box_embedding.universe_box())
        return {
            'unary_probs': vol_U / vol_universe,
            'box_embeddings': U,
        }

class BoxModel(Module):
    def __init__(self, num_models:int, num_boxes:int, dim:int,
                 BoxEmbeddingParam = UnitBoxes, vol_func = clamp_volume):
        super().__init__()
        self.box_embedding = BoxEmbeddingParam(num_models, num_boxes, dim)
        self.unaryprobs = UnaryProbs(self.box_embedding)
        self.condprobs = CondProbs(self.box_embedding)
        self.vol = vol_func
        self.weights = WeightedSum(num_models)

