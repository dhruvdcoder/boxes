from torch import Tensor
import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict, Any, Optional, Union


def _box_shape_ok(t: Tensor) -> bool:
    if len(t.shape) < 2:
        return False
    else:
        if t.size(-2) != 2:
            return False
        return True


def _shape_error_str(tensor_name, expected_shape, actual_shape):
    return "Shape of {} has to be {} but is {}".format(tensor_name,
                                                       expected_shape,
                                                       tuple(actual_shape))


class BoxTensor(Tensor):
    """ Subclass of :class:`Tensor` which has come extra
    box specific methods. It acts as a wrapper to tensor containing
    a collection of boxes or a single box.
    """

    def __new__(cls,
                data: Optional[Tensor] = None,
                requires_grad: bool = False,
                **kwargs: Any) -> "BoxTensor":
        """
        Need to implement new as :class:`torch.Tensor` is a wrapper
        around C.

        Unlike :class:`torch.Tensor`, BoxTensor can take all the arguments
        supported by :func:`torch.tensor` -- dtype=None, device=None, 
        requires_grad=False -- and produce a BoxTensor instance.

        .. todo:: Validate the values of z, Z ? z < Z

        Arguments:

            data: Tensor of shape (**, zZ, num_dims). Here, zZ=2, where
                the 0th dim is for bottom left corner and 1st dim is for
                top right corner of the box

            requires_grad: 
        """
        if data is not None:
            if _box_shape_ok(data):
                val = data
            else:
                raise ValueError(
                    _shape_error_str('data', '(**,2,num_dims)', data.shape))
        else:
            val = torch.Tensor()
        # use 'to()' to set params if specified
        val = val.to(device=kwargs.get('device'), dtype=kwargs.get('dtype'))

        return torch.Tensor._make_subclass(cls, val,
                                           requires_grad)  # type:ignore

    def __repr__(self):
        return 'box_' + super().__repr__()

    def to(self, *args, **kwargs) -> "BoxTensor":
        """ Need to subclass because torch.Tensor.to()
        returns torch.Tensor and not the subclass"""
        res = super().to(*args, **kwargs)
        return self.__class__(res, requires_grad=self.requires_grad)

    def clone(self) -> "BoxTensor":
        """ Need to subclass because torch.Tensor.to()
        returns torch.Tensor and not the subclass"""
        res = super().clone()
        return self.__class__(res, requires_grad=res.requires_grad)

    @property
    def z(self) -> Tensor:
        """Lower left coordinate as Tensor"""
        return self[..., 0, :]

    @property
    def Z(self) -> Tensor:
        """Top right coordinate as Tensor"""
        return self[..., 1, :]

    @classmethod
    def from_zZ(cls, z: Tensor, Z: Tensor) -> "BoxTensor":
        """
        Creates a box by stacking z and Z along -2 dim. 
        That is if z.shape == Z.shape == (**, num_dim),
        then the result would be box of shape (**, 2, num_dim)
        """
        if z.shape != Z.shape:
            raise ValueError(
                "Shape of z and Z should be same but is {} and {}".format(
                    z.shape, Z.shape))
        box_val = torch.stack((z, Z), -2)
        return cls(box_val, requires_grad=(z.requires_grad or Z.requires_grad))

    def intersection(self, other: "BoxTensor") -> "BoxTensor":
        """ Gives intersection of self and other.

        .. note:: This function can give fipped boxes, i.e. where z[i] > Z[i]
        """
        z = torch.max(self.z, other.z)
        Z = torch.min(self.Z, other.Z)
        return self.from_zZ(z, Z)

    def join(self, other: "BoxTensor") -> "BoxTensor":
        """Gives join"""
        z = torch.min(self.z, other.z)
        Z = torch.max(self.Z, other.Z)
        return self.from_zZ(z, Z)

    def get(self, indices: Union[List[int], torch.LongTensor]) -> "BoxTensor":
        """ get boxes at particular indices"""
        return self.__class__(self[..., indices, :])

    def clamp_volume(self) -> Tensor:
        """Volume of boxes. Returns 0 where boxes are flipped.
        
        Returns:

            Tensor of shape (**, ) when self has shape (**, 2, num_dims)
        """
        return torch.prod((self.Z - self.z).clamp_min(0), dim=-1)

    def soft_volume(self, temp: float = 1.) -> Tensor:
        """Volume of boxes. Uses softplus instead of ReLU/clamp

        Returns:

            Tensor of shape (**, ) when self has shape (**, 2, num_dims)
        """
        return torch.prod(F.softplus(self.Z - self.z, beta=temp), dim=-1)

    def log_clamp_volume(self) -> Tensor:
        eps = torch.finfo(self.dtype).tiny  # type: ignore
        res = torch.sum(torch.log((self.Z - self.z).clamp_min(eps)), dim=-1)
        return res

    def log_soft_volume(self, temp: float = 1.) -> Tensor:
        eps = torch.finfo(self.dtype).tiny  # type: ignore
        res = torch.sum(
            torch.log(F.softplus(self.Z - self.z, beta=temp).clamp_min(eps)),
            dim=-1)
        return res
