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
        raise NotImplemented


    def forward(self, box_indices = slice(None, None, None), **kwargs) -> Tensor:
        """
        Returns a Tensor representing the boxes specified by `box_indices` in the form they should be used for training.

        :param box_indices: Slice, List, or Tensor of the box indices
        :param kwargs: Unused for now, but include this for future possible parameters.
        :return: Tensor of shape (model, id, zZ, dim).
        """
        raise NotImplemented

    def universe_box(self) -> Tensor:
        """
        For each model, returns the min/max coordinates of the "universe" box.
            - For boxes which enforce a parametrization in the unit cube, this is simply the unit cube.
            - For boxes which allow a dynamic universe, this is typically the smallest containing box.

        :return: Tensor of shape (model, 1, zZ, dim)
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

    def universe_box(self) -> Tensor:
        """
        In this case, the universe is just the [0,1] hypercube.
        :return: Tensor of shape (model, 1, zZ, dim) representing [0,1]^d
        """
        z = torch.zeros(self.boxes.shape[0], 1, self.boxes.shape[-1])
        Z = torch.ones(self.boxes.shape[0], 1, self.boxes.shape[-1])
        return torch.stack((z, Z), dim=2).to(self.boxes.device)


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


    def universe_box(self) -> Tensor:
        """
        In this case, we calculate the smallest containing box.
        :return: Tensor of shape (model, 1, zZ, dim)
        """
        min_z, _ = torch.min(self.z, dim=1, keepdim=True)
        max_Z, _ = torch.max(torch.exp(self.logdelta), dim=1, keepdim=True)
        return torch.stack((min_z, max_Z), dim=2)


###############################################
# Downstream Model
###############################################

class CondProbs(Module):
    def __init__(self, box_embedding: BoxEmbedding, vol_func = clamp_volume):
        super().__init__()
        self.box_embedding = box_embedding
        self.vol_func = vol_func

    def forward(self, box_indices_A: Tensor, box_indices_B: Tensor) -> Dict:
        A = self.box_embedding(box_indices_A)
        B = self.box_embedding(box_indices_B)
        vol_A_int_B = self.vol_func(intersection(A, B))
        vol_A = self.vol_func(A)
        return {
            "A": A,
            "B": B,
            "P(B|A)": vol_A_int_B / vol_A,
        }


class UnaryProbs(Module):

    def __init__(self, box_embedding, vol_func: Callable = clamp_volume):
        super().__init__()
        self.box_embedding = box_embedding
        self.vol_func = vol_func

    def forward(self):
        U = self.box_embedding()
        vol_U = self.vol_func(U)
        vol_universe = self.vol_func(self.box_embedding.universe_box())
        return {
            "unary_probs": vol_U / vol_universe,
            "box_embeddings": U,
        }


class WeightedSum(Module):
    def __init__(self, num_models: int) -> None:
        super().__init__()
        self.weights = Parameter(torch.rand(num_models))

    def forward(self, box_vols: Tensor) -> Tensor:
        return (F.softmax(self.weights, dim=0).unsqueeze(0) @ box_vols).squeeze()


class BoxModel(Module):
    def __init__(self, num_models:int, num_boxes:int, dim:int,
                 BoxEmbeddingParam: type, vol_func: Callable):
        super().__init__()
        self.box_embedding = BoxEmbeddingParam(num_models, num_boxes, dim)
        self.unary_probs = UnaryProbs(self.box_embedding, vol_func)
        self.cond_probs = CondProbs(self.box_embedding, vol_func)
        self.vol_func = vol_func
        self.weights = WeightedSum(num_models)

    def forward(self, box_indices: Tensor) -> Dict:
        unary = self.unary_probs()
        cond = self.cond_probs(box_indices[:,0], box_indices[:,1])
        return {
            "unary_probs": self.weights(unary["unary_probs"]),
            "box_embeddings": unary["box_embeddings"],
            "A": cond["A"],
            "B": cond["B"],
            "P(B|A)": self.weights(cond["P(B|A)"]),
        }
