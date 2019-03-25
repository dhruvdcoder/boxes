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
    An example class for creating a Box Embedding parametrization.
    Don't inherit from this, it is just an example which contains the methods for a class to be used as a BoxEmbedding
    layer. Refer to the docstring of the functions when implementing your own BoxEmbedding.

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
        raise NotImplemented


    def forward(self, box_indices = slice(None, None, None), **kwargs) -> Tensor:
        """
        Returns a Tensor representing the boxes specified by `box_indices` in the form they should be used for training.

        :param box_indices: Slice, List, or Tensor of the box indices
        :param kwargs: Unused for now, but include this for future possible parameters.
        :return: Tensor of shape (model, id, zZ, dim).
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
        rand_param = lambda min, max: min + torch.rand(num_models, num_boxes, dim) * max
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

        self.boxes = Parameter(torch.stack((z, Z), dim=2))

    def forward(self, box_indices = slice(None, None, None), **kwargs) -> Tensor:
        """
        Returns a Tensor representing the box embeddings specified by box_indices.

        :param box_indices: Slice, List, or Tensor of the box indices
        :param kwargs: Unused for now, but include this for future possible parameters.
        :return: NamedTensor of shape (model, id, zZ, dim).
        """
        return self.boxes[:, box_indices]


class MinMaxUnitBoxes(Module):
    """
    Parametrize boxes in \RR^d by using 2d coordinates.

    self.boxes[model, box, 2, dim] \in [0,1]

    In this parametrization, we select the z/Z coordinates simply by
    taking the min/max over axis 2, i.e.

    z, _ = torch.min(self.boxes, dim=2) # Tensor of shape (model, box, dim)
    Z, _ = torch.max(self.boxes, dim=2) # Tensor of shape (model, box, dim)

    This avoids the need to make sure the boxes don't "flip", i.e. Z becomes smaller than z.
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
        rand_param = lambda min, max: min + torch.rand(num_models, num_boxes, dim) * max
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

        self.boxes = Parameter(torch.stack((z, Z), dim=2))

    def forward(self, box_indices = slice(None, None, None), **kwargs) -> Tensor:
        """
        Returns a Tensor representing the box embeddings specified by box_indices.

        :param box_indices: Slice, List, or Tensor of the box indices
        :param kwargs: Unused for now, but include this for future possible parameters.
        :return: NamedTensor of shape (model, id, zZ, dim).
        """
        z, _ = torch.min(self.boxes[:, box_indices], dim=2) # Tensor of shape (model, box, dim)
        Z, _ = torch.max(self.boxes[:, box_indices], dim=2) # Tensor of shape (model, box, dim)
        return torch.stack((z, Z), dim=2)


class DeltaBoxes(Module):
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


    def forward(self, box_indices = slice(None, None, None), **kwargs) -> Tensor:
        """
        Returns a Tensor representing the box embeddings specified by box_indices.

        :param box_indices: A NamedTensor of the box indices
        :param kwargs: Unused for now, but include this for future possible parameters.
        :return: Tensor of shape (model, id, zZ, dim).
        """
        return torch.stack((self.z[:, box_indices], self.z[:, box_indices] + torch.exp(self.logdelta[:, box_indices])), dim=2)


class SigmoidBoxes(Module):
    """
    Parametrize boxes using sigmoid to make them always valid and contained within the unit cube.

    self.w[model, box, dim] \in \RR
    self.W[model, box, dim] \in \RR

    z = sigmoid(w)
    Z = z + sigmoid(W) * (1-z)

    This forces z \in (0,1), Z \in (z, 1).

    NOT IMPLEMENTED YET:
    Optionally, indicate an eps value, such that side lengths can get no smaller than eps.
    In this case, we would have

    z = sigmoid(w)
    Z = (z + eps) + sigmoid(W)*(1-(z+eps))

    and the minimum volume of a box would be eps^d.
    """

    def __init__(self, num_models: int, num_boxes: int, dim: int, **kwargs):
        super().__init__()
        self.w = Parameter(torch.rand(num_models, num_boxes, dim))
        self.W = Parameter(torch.rand(num_models, num_boxes, dim))


    def forward(self, box_indices = slice(None, None, None), **kwargs) -> Tensor:
        """
        Returns a Tensor representing the box embeddings specified by box_indices.

        :param box_indices: A NamedTensor of the box indices
        :param kwargs: Unused for now, but include this for future possible parameters.
        :return: Tensor of shape (model, id, zZ, dim).
        """
        z = F.sigmoid(self.w[:, box_indices])
        Z = z + F.sigmoid(self.W[:, box_indices]) * (1-z)
        return torch.stack((z,Z), dim=2)


###############################################
# Downstream Model
###############################################

class WeightedSum(Module):
    def __init__(self, num_models: int) -> None:
        super().__init__()
        self.weights = Parameter(torch.rand(num_models))

    def forward(self, box_vols: Tensor) -> Tensor:
        return (F.softmax(self.weights, dim=0).unsqueeze(0) @ box_vols).squeeze()


class BoxModel(Module):
    def __init__(self, num_models:int, num_boxes:int, dim:int,
                 BoxEmbeddingParam: type, vol_func: Callable,
                 universe_box: Callable = None):
        super().__init__()
        self.box_embedding = BoxEmbeddingParam(num_models, num_boxes, dim)
        self.vol_func = vol_func

        if universe_box is None:
            z = torch.zeros(dim)
            Z = torch.ones(dim)
            self.universe_box = lambda _: torch.stack((z,Z))[None, None]
            self.universe_vol = lambda _: self.vol_func(self.universe_box(None)).squeeze()
            self.clamp = True
        else:
            self.universe_box = universe_box
            self.universe_vol = lambda b: self.vol_func(self.universe_box(b))
            self.clamp = False

        self.weights = WeightedSum(num_models)

    def forward(self, box_indices: Tensor) -> Dict:
        # Unary
        box_embeddings_orig = self.box_embedding()
        if self.clamp:
            box_embeddings = box_embeddings_orig.clamp(0,1)
        else:
            box_embeddings = box_embeddings_orig

        unary_probs = self.vol_func(box_embeddings)

        # Conditional
        A = box_embeddings[:, box_indices[:,0]]
        B = box_embeddings[:, box_indices[:,1]]
        P_B_given_A = self.vol_func(intersection(A, B)) / (unary_probs[:, box_indices[:,0]] + 1e-38)

        # Scale Unary
        unary_probs = unary_probs / self.universe_vol(box_embeddings)

        return {
            "unary_probs": self.weights(unary_probs),
            "box_embeddings": box_embeddings_orig,
            "A": A,
            "B": B,
            "P(B|A)": self.weights(P_B_given_A),
        }
