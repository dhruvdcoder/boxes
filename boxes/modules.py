import torch
from torch import Tensor
from torch.nn import Module, Parameter
import torch.nn.functional as F
from .box_operations import *
from .utils import *


################################################
# Box Parametrization Layers
################################################
default_init_min_vol = torch.finfo(torch.float32).tiny

class BoxParam(Module):
    """
    An example class for creating a box parametrization.
    Don't inherit from this, it is just an example which contains the methods for a class to be used as a BoxParam
    layer. Refer to the docstring of the functions when implementing your own BoxParam.

    Note: to avoid naming conflicts with min/max functions, we refer to the min coordinate for a box as `z`, and the
    max coordinate as `Z`.
    """

    def __init__(self, num_models:int, num_boxes:int, dim:int, init_min_vol:float = default_init_min_vol, **kwargs):
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


class Boxes(Module):
    """
    Parametrize boxes using the min coordinate and max coordinate,
    initialized to be in the unit hypercube.

    self.boxes[model, box, min/max, dim] \in [0,1]

    In this parametrization, the min and max coordinates are explicitly stored
    in separate dimensions (as shown above), which means that care must be
    taken to preserve max > min while training. (See MinBoxSize Callback.)
    """

    def __init__(self, num_models: int, num_boxes: int, dims: int, init_min_vol: float = default_init_min_vol, **kwargs):
        """
        Creates the Parameters used for the representation of boxes.
        Initializes boxes with a uniformly random distribution of coordinates, ensuring that each box
        contains a cube of volume larger than init_min_vol.

        :param num_models: Number of models
        :param num_boxes: Number of boxes
        :param dims: Dimension
        :param init_min_vol: Minimum volume for boxes which are created
        :param kwargs: Unused for now, but include this for future possible parameters.
        """
        super().__init__()
        rand_param = lambda min, max: min + torch.rand(num_models, num_boxes, dims) * (max - min)
        if init_min_vol == 0:
            per_dim_min = 0
        elif init_min_vol > 0:
            per_dim_min = torch.tensor(init_min_vol).pow(1/dims)
        else:
            raise ValueError(f"init_min_vol={init_min_vol} is an invalid option.")

        z = rand_param(0, 1-per_dim_min)
        Z = rand_param(z+per_dim_min, 1)

        self.boxes = Parameter(torch.stack((z, Z), dim=2))

    def forward(self, box_indices = slice(None, None, None), **kwargs) -> Tensor:
        """
        Returns a Tensor representing the box embeddings specified by box_indices.

        :param box_indices: Slice, List, or Tensor of the box indices
        :param kwargs: Unused for now, but include this for future possible parameters.
        :return: NamedTensor of shape (model, id, zZ, dim).
        """
        return self.boxes[:, box_indices]


class MinMaxBoxes(Module):
    """
    Parametrize boxes in \RR^d by using 2d coordinates.

    self.boxes[model, box, 2, dim] \in [0,1]

    In this parametrization, we select the z/Z coordinates simply by
    taking the min/max over axis 2, i.e.

    z, _ = torch.min(self.boxes, dim=2) # Tensor of shape (model, box, dim)
    Z, _ = torch.max(self.boxes, dim=2) # Tensor of shape (model, box, dim)

    This avoids the need to make sure the boxes don't "flip", i.e. Z becomes smaller than z.
    """

    def __init__(self, num_models: int, num_boxes: int, dim: int, init_min_vol: float = default_init_min_vol, **kwargs):
        """
        Creates the Parameters used for the representation of boxes.

        :param num_models: Number of models
        :param num_boxes: Number of boxes
        :param dim: Dimension
        :param init_min_vol: Creates boxes which a cube of this volume.
        :param kwargs: Unused for now, but include this for future possible parameters.
        """
        super().__init__()
        unit_boxes = Boxes(num_models, num_boxes, dim, init_min_vol)
        self.boxes = Parameter(unit_boxes.boxes.detach().clone())
        del unit_boxes

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

    def __init__(self, num_models: int, num_boxes: int, dim: int, init_min_vol: float = default_init_min_vol, **kwargs):
        super().__init__()
        unit_boxes = Boxes(num_models, num_boxes, dim, init_min_vol)
        self._from_UnitBoxes(unit_boxes)
        del unit_boxes

    def _from_UnitBoxes(self, unit_boxes:Boxes):
        boxes = unit_boxes.boxes.detach().clone()
        z = boxes[:,:,0]
        Z = boxes[:,:,1]
        self.z = Parameter(z)
        self.logdelta = Parameter(torch.log(Z-z))

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

    self.w[model, box, dim] in Reals
    self.W[model, box, dim] in Reals

    z = sigmoid(w)
    Z = z + sigmoid(W) * (1-z)

    This forces z in (0,1), Z in (z, 1).
    """

    def __init__(self, num_models: int, num_boxes: int, dim: int, init_min_vol: float = default_init_min_vol,  **kwargs):
        super().__init__()
        unit_boxes = Boxes(num_models, num_boxes, dim, init_min_vol)
        self._from_UnitBoxes(unit_boxes)
        del unit_boxes


    def _from_UnitBoxes(self, unit_boxes:Boxes):
        boxes = unit_boxes().detach().clone()
        z = boxes[:,:,0]
        Z = boxes[:,:,1]
        l = (Z-z) / (1-z)
        self.w = Parameter(torch.log(z / (1-z)))
        self.W = Parameter(torch.log(l / (1-l)))


    def forward(self, box_indices = slice(None, None, None), **kwargs) -> Tensor:
        """
        Returns a Tensor representing the box embeddings specified by box_indices.

        :param box_indices: A NamedTensor of the box indices
        :param kwargs: Unused for now, but include this for future possible parameters.
        :return: Tensor of shape (model, id, zZ, dim).
        """
        z = torch.sigmoid(self.w[:, box_indices])
        Z = z + torch.sigmoid(self.W[:, box_indices]) * (1-z)
        return torch.stack((z,Z), dim=2)


class MinMaxSigmoidBoxes(Module):
    """
    Parametrize boxes using sigmoid to make them always valid and contained within the unit cube.

    self.boxes[model, box, 2, dim] in Reals


    In this parametrization, we first convert to the unit cube:

    unit_cube_boxes = torch.sigmoid(self.boxes)  # shape: (model, box, 2, dim)

    We now select the z/Z coordinates by taking the min/max over axis 2, i.e.

    z, _ = torch.min(unit_cube_boxes, dim=2)
    Z, _ = torch.max(unit_cube_boxes, dim=2)
    """

    def __init__(self, num_models: int, num_boxes: int, dim: int, init_min_vol: float = default_init_min_vol,  **kwargs):
        super().__init__()
        unit_boxes = Boxes(num_models, num_boxes, dim, init_min_vol)
        self._from_UnitBoxes(unit_boxes)
        del unit_boxes


    def _from_UnitBoxes(self, unit_boxes:Boxes):
        boxes = unit_boxes().detach().clone()
        self.boxes = Parameter(torch.log(boxes / (1-boxes)))


    def forward(self, box_indices = slice(None, None, None), **kwargs) -> Tensor:
        """
        Returns a Tensor representing the box embeddings specified by box_indices.

        :param box_indices: A NamedTensor of the box indices
        :param kwargs: Unused for now, but include this for future possible parameters.
        :return: Tensor of shape (model, id, zZ, dim).
        """
        unit_cube_boxes = torch.sigmoid(self.boxes)
        z, _ = torch.min(unit_cube_boxes, dim=2)
        Z, _ = torch.max(unit_cube_boxes, dim=2)
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
    def __init__(self, BoxParamType: type, vol_func: Callable,
                 num_models:int, num_boxes:int, dims:int,
                 init_min_vol: float = default_init_min_vol, universe_box: Optional[Callable] = None):
        super().__init__()
        self.box_embedding = BoxParamType(num_models, num_boxes, dims, init_min_vol)
        self.vol_func = vol_func

        if universe_box is None:
            z = torch.zeros(dims)
            Z = torch.ones(dims)
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
        P_B_given_A = self.vol_func(intersection(A, B)) / unary_probs[:, box_indices[:,0]] # + 1e-38)

        # Scale Unary
        unary_probs = unary_probs / self.universe_vol(box_embeddings)

        return {
            "unary_probs": self.weights(unary_probs),
            "box_embeddings": box_embeddings_orig,
            "A": A,
            "B": B,
            "P(B|A)": self.weights(P_B_given_A),
        }
