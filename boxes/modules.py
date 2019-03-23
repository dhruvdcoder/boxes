import torch
from torch import Tensor
from namedtensor import ntorch, NamedTensor
from namedtensor.nn.torch_nn import Module
import torch.nn.functional as F
from .box_operations import *

from torch.nn import Parameter

################################################
# Box Embedding Layers
################################################


class BoxEmbedding(Module):
    """
    An example class for creating a Box Embedding parametrization.
    Don't inherit from this, it is just an example which contains the minimum required methods to be used as a
    box embedding. At a minimum, you should define methods with the same signature as those that this class contains.

    Note: to avoid naming conflicts with min/max functions, we refer to the min coordinate for a box as `z`, and the
    max coordinate as `Z`.
    """

    def __init__(self, num_models:int, num_boxes:int, dim:int, **kwargs):
        """
        Creates the Parameters used for the representation of boxes.

        :param num_models: Number of models
        :param num_boxes: Number of boxes
        :param dim: Dimension
        :param kwargs: Unused for now, but include this for future possible parameters.
        """
        # Remember to call:
        super().__init__()
        pass


    def forward(self, box_indices: Union[NamedTensor, None] = None, **kwargs) -> NamedTensor:
        """
        Returns a Tensor representing the boxes specified by `box_indices` in the form they should be used for training.

        :param box_indices: A NamedTensor of the box indices
        :param kwargs: Unused for now, but include this for future possible parameters.
        :return: NamedTensor of shape (model, id, zZ, dim).
        """
        raise NotImplemented

    def universe_box(self) -> Tensor:
        """
        For each model, returns the min/max coordinates of the "universe" box.
            - For boxes which enforce a parametrization in the unit cube, this is simply the unit cube.
            - For boxes which allow a dynamic universe, this is typically the smallest containing box.

        :return: NamedTensor of shape (model, zZ, dim)
        """
        raise NotImplemented


class UnitBoxes(Module):
    """
    Parametrize boxes using the min coordinate and max coordinate,
    initialized to be in the unit hypercube.

    self.boxes[model, box, min/max, dim] \in [0,1]

    In this parametrization, the min and max coordinates are explicitly stored
    in separate dimensions (as shown above), which means that care must be
    taken to preserve max > min while training. (See MinBoxSize Callback.)
    """

    def __init__(self, num_models: int, num_boxes: int, dim: int, init_min_vol: float = 1e-2, **kwargs):
        """
        Creates the Parameters used for the representation of boxes.

        :param num_models: Number of models
        :param num_boxes: Number of boxes
        :param dim: Dimension
        :param init_min_vol: Creates boxes which a cube of this volume.
        :param kwargs: Unused for now, but include this for future possible parameters.
        """
        super().__init__()
        rand_param = lambda min, max: min + ntorch.rand(num_models, num_boxes, dim, names=("model", "box", "dim")) * max
        if init_min_vol == 0:
            """
            Uniformly random distribution of coordinates, ensuring
                Z_j > z_j for all j.
            """
            z = rand_param(0, 1)
            Z = rand_param(z, 1-z)
        elif init_min_vol > 0:
            """
            Uniformly random distribution of coordinates, ensuring that each box
            contains a cube of volume larger than init_min_vol.
            Useful in higher dimensions to deal with underflow, however in high
            dimensions a cube with small volume still has very large side-lengths.
            """
            eps = init_min_vol ** (1 / dim)
            z = rand_param(0, 1-eps)
            Z = rand_param(z+eps, 1-(z+eps))
        else:
            raise ValueError(f"init_min_vol={init_min_vol} is an invalid option.")

        self.boxes = ntorch.stack((z,Z), "zZ")
        # We have to use self.register_parameter instead of Paramter because namedtensor doesn't currently support the latter.
        self.register_parameter("boxes", self.boxes)

    def forward(self, box_indices: Union[NamedTensor, None] = None, **kwargs):
        """
        Returns a Tensor representing the box embeddings specified by box_indices.

        :param box_indices: A NamedTensor of the box indices
        :param kwargs: Unused for now, but include this for future possible parameters.
        :return: NamedTensor of shape (model, id, zZ, dim).
        """
        # We have to use this branching condition because namedtensor doesn't currently support slice(None, None, None)
        if box_indices is None:
            return self.boxes
        else:
            return self.boxes[{"box": box_indices}]

    def universe_box(self):
        """
        In this case, the universe is just the [0,1] hypercube.
        :return: NamedTensor of shape (model, zZ, dim) representing [0,1]^d
        """
        named_dims = self.boxes.shape
        z = ntorch.zeros(named_dims["model"], named_dims["dim"])
        Z = ntorch.ones(named_dims["model"], named_dims["dim"])
        return ntorch.stack((z,Z), name="zZ")


class DeltaBoxes(Module):
    """
    Parametrize boxes using the min coordinate and log of side-length.

    self.z[model, box, dim] \in \RR
    self.logdelta[model, box, dim] \in \RR

    This forces boxes to always have positive side-lengths.
    """

    def __init__(self, num_models: int, num_boxes: int, dim: int, **kwargs):
        super().__init__()
        self.z = ntorch.rand(num_models, num_boxes, dim, names=("model", "box", "dim"))
        self.logdelta = ntorch.rand(num_models, num_boxes, dim, names=("model", "box", "dim"))
        self.register_parameter("z", self.z)
        self.register_parameter("logdelta", self.logdelta)


    def forward(self, box_indices: Union[NamedTensor, None] = None, **kwargs):
        """
        Returns a Tensor representing the box embeddings specified by box_indices.

        :param box_indices: A NamedTensor of the box indices
        :param kwargs: Unused for now, but include this for future possible parameters.
        :return: NamedTensor of shape (model, id, zZ, dim).
        """
        # We have to use this branching condition because namedtensor doesn't currently support slice(None, None, None)
        if box_indices is None:
            return ntorch.stack((self.z, self.z + self.Z()), "zZ")
        else:
            return ntorch.stack((self.z[{"box": box_indices}], self.z[{"box": box_indices}] + self.Z(box_indices)), "zZ")

    def Z(self, box_indices: Union[NamedTensor, None] = None, **kwargs):
        """
        Returns a Tensor representing the max coordinate of boxes specified by ids as they should be used for training.

        :param box_indices: A NamedTensor of the box indices
        :param kwargs: Unused for now, but include this for future possible parameters.
        :return: NamedTensor of shape (model, id, dim).
        """
        if box_indices is None:
            return self.z + ntorch.exp(self.logdelta)
        else:
            return self.z[{"box": box_indices}] + ntorch.exp(self.logdelta[{"box": box_indices}])


    def universe_box(self):
        """
        In this case, the universe is just the [0,1] hypercube.
        :return: NamedTensor of shape (model, zZ, dim) representing [0,1]^d
        """
        named_dims = self.boxes.shape
        z = ntorch.zeros(named_dims["model"], named_dims["dim"])
        Z = ntorch.ones(named_dims["model"], named_dims["dim"])
        return ntorch.stack((z,Z), name="zZ")


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

