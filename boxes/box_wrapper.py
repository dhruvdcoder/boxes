from torch import Tensor
import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict, Any, Optional, Union, Type, TypeVar
from boxes.utils import log1mexp
tanh_eps = 1e-20


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


# see: https://realpython.com/python-type-checking/#type-hints-for-methods
# to know why we need to use TypeVar
TBoxTensor = TypeVar("TBoxTensor", bound="BoxTensor")


class BoxTensor(object):
    """ A wrapper to which contains single tensor which
    represents single or multiple boxes.

    Have to use composition instead of inheritance because
    it is not safe to interit from :class:`torch.Tensor` because
    creating an instance of such a class will always make it a leaf node.
    This works for :class:`torch.nn.Parameter` but won't work for a general
    box_tensor.
    """

    def __init__(self, data: Tensor) -> None:
        """
        .. todo:: Validate the values of z, Z ? z < Z

        Arguments:
            data: Tensor of shape (**, zZ, num_dims). Here, zZ=2, where
                the 0th dim is for bottom left corner and 1st dim is for
                top right corner of the box
        """

        if _box_shape_ok(data):
            self.data = data
        else:
            raise ValueError(
                _shape_error_str('data', '(**,2,num_dims)', data.shape))
        super().__init__()

    def __repr__(self):
        return 'box_tensor_wrapper(' + self.data.__repr__() + ')'

    @property
    def z(self) -> Tensor:
        """Lower left coordinate as Tensor"""

        return self.data[..., 0, :]

    @property
    def Z(self) -> Tensor:
        """Top right coordinate as Tensor"""

        return self.data[..., 1, :]

    @classmethod
    def from_zZ(cls: Type[TBoxTensor], z: Tensor, Z: Tensor) -> TBoxTensor:
        """
        Creates a box by stacking z and Z along -2 dim.
        That is if z.shape == Z.shape == (**, num_dim),
        then the result would be box of shape (**, 2, num_dim)
        """

        if z.shape != Z.shape:
            raise ValueError(
                "Shape of z and Z should be same but is {} and {}".format(
                    z.shape, Z.shape))
        box_val: Tensor = torch.stack((z, Z), -2)

        return cls(box_val)

    @classmethod
    def from_split(cls: Type[TBoxTensor], t: Tensor,
                   dim: int = -1) -> TBoxTensor:
        """Creates a BoxTensor by splitting on the dimension dim at midpoint

        Args:
            t: input
            dim: dimension to split on

        Returns:
            BoxTensor: output BoxTensor

        Raises:
            ValueError: `dim` has to be even
        """
        len_dim = t.size(dim)

        if len_dim % 2 != 0:
            raise ValueError(
                "dim has to be even to split on it but is {}".format(
                    t.size(dim)))
        split_point = int(len_dim / 2)
        z = t.index_select(
            dim,
            torch.tensor(
                list(range(split_point)), dtype=torch.int64, device=t.device))

        Z = t.index_select(
            dim,
            torch.tensor(
                list(range(split_point, len_dim)),
                dtype=torch.int64,
                device=t.device))

        return cls.from_zZ(z, Z)

    def _intersection(self: TBoxTensor,
                      other: TBoxTensor) -> Tuple[Tensor, Tensor]:
        z = torch.max(self.z, other.z)
        Z = torch.min(self.Z, other.Z)

        return z, Z

    def intersection(self: TBoxTensor, other: TBoxTensor) -> TBoxTensor:
        """ Gives intersection of self and other.

        .. note:: This function can give fipped boxes, i.e. where z[i] > Z[i]
        """
        z, Z = self._intersection(other)

        return self.from_zZ(z, Z)

    def join(self: TBoxTensor, other: TBoxTensor) -> TBoxTensor:
        """Gives join"""
        z = torch.min(self.z, other.z)
        Z = torch.max(self.Z, other.Z)

        return self.from_zZ(z, Z)

    def get(self: TBoxTensor, indices: torch.LongTensor,
            dim: int = 0) -> TBoxTensor:
        """ Get boxes at particular indices on a particular dimension.

        Shape of indices should be
        according to the shape of BoxTensor. For instance, if shape of
        BoxTensor is (3,4,2,5), then shape of indice should be (*,*)

        """

        return self.__class__(self.data.index_select(dim, indices))

    def clamp_volume(self) -> Tensor:
        """Volume of boxes. Returns 0 where boxes are flipped.

        Returns:

            Tensor of shape (**, ) when self has shape (**, 2, num_dims)
        """

        return torch.prod((self.Z - self.z).clamp_min(0), dim=-1)

    @classmethod
    def _in_zero_one(cls, t: Union[Tensor, float]):
        if torch.is_tensor(t):
            if not ((t > 0.) * (t <= 1.)).all():
                return False
            else:
                return True

        if isinstance(t, float):
            return (0. < t <= 1.)

    @classmethod
    def _soft_volume(cls,
                     z: Tensor,
                     Z: Tensor,
                     temp: float = 1.,
                     scale: Union[float, Tensor] = 1.) -> Tensor:
        """ scale has to be between 0 and 1"""

        if not cls._in_zero_one(scale):
            raise ValueError(
                "Scale should be in (0,1] but is {}".format(scale))
        side_lengths = (F.softplus(Z - z, beta=temp))

        return torch.prod(side_lengths, dim=-1) * scale

    def soft_volume(self, temp: float = 1.,
                    scale: Union[float, Tensor] = 1.) -> Tensor:
        """Volume of boxes. Uses softplus instead of ReLU/clamp

        Returns:
            Tensor of shape (**, ) when self has shape (**, 2, num_dims)
        """

        return self._soft_volume(self.z, self.Z, temp, scale)

    def intersection_soft_volume(self,
                                 other: TBoxTensor,
                                 temp: float = 1.,
                                 scale: Union[float, Tensor] = 1.) -> Tensor:
        """ Computes the soft volume of the intersection box

        Return:
            Tensor of shape(**,) when self and other have shape (**, 2, num_dims)
        """
        # intersection
        z, Z = self._intersection(other)

        return self._soft_volume(z, Z, temp, scale)

    def log_clamp_volume(self) -> Tensor:
        eps = torch.finfo(self.data.dtype).tiny  # type: ignore
        res = torch.sum(torch.log((self.Z - self.z).clamp_min(eps)), dim=-1)

        return res

    @classmethod
    def _log_soft_volume(cls,
                         z: Tensor,
                         Z: Tensor,
                         temp: float = 1.,
                         scale: Union[float, Tensor] = 1.) -> Tensor:
        eps = torch.finfo(z.dtype).tiny  # type: ignore

        if isinstance(scale, float):
            s = torch.tensor(scale)
        else:
            s = scale

        return (torch.sum(
            torch.log(F.softplus(Z - z, beta=temp).clamp_min(eps)),
            dim=-1) + torch.log(s)
        )  # need this eps to that the derivative of log does not blow

    def log_soft_volume(self,
                        temp: float = 1.,
                        scale: Union[float, Tensor] = 1.) -> Tensor:
        res = self._log_soft_volume(self.z, self.Z, temp=temp, scale=scale)

        return res

    def intersection_log_soft_volume(
            self,
            other: TBoxTensor,
            temp: float = 1.,
            scale: Union[float, Tensor] = 1.) -> Tensor:
        z, Z = self._intersection(other)
        vol = self._log_soft_volume(z, Z, temp=temp, scale=scale)

        return vol

    @classmethod
    def _log_conditional_prob(
            cls,
            box1: TBoxTensor,
            box2: TBoxTensor,
            temp: float = 1.,
            scale: Union[float, Tensor] = 1.) -> Tuple[Tensor, Tensor]:
        """ Gives P(b1|b2=1) ie two values, one for b1=1 and other for b1=0

        Assume the shape of boxes to be (**, 2, num_dim)
        """
        log_numerator = box1.intersection_log_soft_volume(
            box2, temp=temp)  # shape = (**,)
        log_denominator = box2.log_soft_volume(temp=temp)  # shape =(**,)

        if not cls._in_zero_one(scale):
            raise ValueError(
                "scale should be in (0,1] but is {}".format(scale))

        if isinstance(scale, float):
            s = torch.tensor(scale)
        else:
            s = scale

        log_cp1 = log_numerator - log_denominator + torch.log(s)
        log_cp2 = log1mexp(log_cp1)

        return log_cp1, log_cp2

    def log_conditional_prob(
            self: TBoxTensor,
            on_box: TBoxTensor,
            temp: float = 1.,
            scale: Union[float, Tensor] = 1.) -> Tuple[Tensor, Tensor]:

        return self._log_conditional_prob(self, on_box, temp=temp, scale=scale)

    @classmethod
    def cat(cls: Type[TBoxTensor],
            tensors: Tuple[TBoxTensor, ...]) -> TBoxTensor:

        return cls(torch.cat(tuple(map(lambda x: x.data, tensors)), -1))

    @classmethod
    def _scaled_box(cls, z_F: Tensor, Z_F: Tensor, z_R: Tensor,
                    Z_R: Tensor) -> Tuple[Tensor, Tensor]:
        L_R = (Z_R - z_R).clamp_min(0)
        z_S = z_R + z_F * L_R
        Z_S = Z_R + (Z_F - 1) * L_R

        return z_S, Z_S

    def scaled_box(self, ref_box: TBoxTensor) -> "BoxTensor":
        z, Z = self._scaled_box(self.z, self.Z, ref_box.z, ref_box.Z)

        return BoxTensor.from_zZ(z, Z)


def inv_sigmoid(v: Tensor) -> Tensor:
    return torch.log(v / (1. - v))  # type:ignore


class SigmoidBoxTensor(BoxTensor):
    """Same as BoxTensor but with a different parameterization: (**,wW, num_dims)

    z = sigmoid(w)
    Z = z + sigmoid(W) * (1-z)

    w = inv_sigmoid(z)
    W = inv_sigmoid((Z - z)/(1-z))
    """

    @property
    def z(self) -> Tensor:
        return torch.sigmoid(self.data[..., 0, :])

    @property
    def Z(self) -> Tensor:
        z = self.z
        Z = z + torch.sigmoid(self.data[..., 1, :]) * (1. - z)  # type: ignore

        return Z

    @classmethod
    def from_zZ(cls: Type[TBoxTensor], z: Tensor, Z: Tensor) -> TBoxTensor:
        if z.shape != Z.shape:
            raise ValueError(
                "Shape of z and Z should be same but is {} and {}".format(
                    z.shape, Z.shape))
        eps = torch.finfo(z.dtype).tiny  # type: ignore
        w = inv_sigmoid(z.clamp(eps, 1. - eps))
        W = inv_sigmoid(((Z - z) / (1. - z)).clamp(eps,
                                                   1. - eps))  # type:ignore

        box_val: Tensor = torch.stack((w, W), -2)

        return cls(box_val)

    @classmethod
    def from_split(cls: Type[TBoxTensor], t: Tensor,
                   dim: int = -1) -> TBoxTensor:
        """Creates a BoxTensor by splitting on the dimension dim at midpoint

        Args:
            t: input
            dim: dimension to split on

        Returns:
            BoxTensor: output BoxTensor

        Raises:
            ValueError: `dim` has to be even
        """
        len_dim = t.size(dim)

        if len_dim % 2 != 0:
            raise ValueError(
                "dim has to be even to split on it but is {}".format(
                    t.size(dim)))
        split_point = int(len_dim / 2)
        w = t.index_select(
            dim,
            torch.tensor(
                list(range(split_point)), dtype=torch.int64, device=t.device))

        W = t.index_select(
            dim,
            torch.tensor(
                list(range(split_point, len_dim)),
                dtype=torch.int64,
                device=t.device))
        box_val: Tensor = torch.stack((w, W), -2)

        return cls(box_val)


class TanhActivatedBoxTensor(BoxTensor):
    """Same as BoxTensor but with a parameterization which is assumed to be the output
    from an activation function.

    Supported activations:

        1. tanh

    let (*, num_dims) be the shape of output of the activations, then the BoxTensor is
    created with shape (*, zZ, num_dims/2)

    For tanh:

    z = (w + 1)/2

    Z = z + ((W + 1)/2) * (1-z)
    => To avoid zero volume boxes z should not be equal to Z=> w should be in [-1., 1.)
    => Also, W cannot be -1 => W should be in (-1, 1]

    where w and W are outputs of tanh and hence are in (-1, 1)

    => 0 < z < 1

    => z < Z < 1

    w = 2z -1
    W = 2(Z - z)/(1-z) -1
    """

    @classmethod
    def w2z(cls, w: torch.Tensor) -> torch.Tensor:
        return (w + 1) / 2

    @property
    def z(self) -> Tensor:
        return self.w2z(self.data[..., 0, :])

    @property
    def Z(self) -> Tensor:
        z = self.z
        Z = z + self.w2z(self.data[..., 1, :]) * (1. - z)  # type: ignore

        return Z

    @classmethod
    def from_zZ(cls: Type[TBoxTensor], z: Tensor, Z: Tensor) -> TBoxTensor:
        if z.shape != Z.shape:
            raise ValueError(
                "Shape of z and Z should be same but is {} and {}".format(
                    z.shape, Z.shape))
        z_ = z.clamp(0., 1. - tanh_eps / 2.)
        Z_ = Z.clamp(tanh_eps / 2., 1.)
        w = (2 * z_ - 1)
        W = 2 * (Z_ - z_) / (1. - z_) - 1.

        box_val: Tensor = torch.stack((w, W), -2)

        return cls(box_val)

    def intersection(self: TBoxTensor, other: TBoxTensor) -> TBoxTensor:
        """ Gives intersection of self and other.

        .. note:: This function can give fipped boxes, i.e. where z[i] > Z[i]
        """
        z, Z = self._intersection(other)

        return BoxTensor.from_zZ(z, Z)

    @classmethod
    def from_split(cls: Type[TBoxTensor], t: Tensor,
                   dim: int = -1) -> TBoxTensor:
        """Creates a BoxTensor by splitting on the dimension dim at midpoint

        Args:
            t: input
            dim: dimension to split on

        Returns:
            BoxTensor: output BoxTensor

        Raises:
            ValueError: `dim` has to be even
        """
        len_dim = t.size(dim)

        if len_dim % 2 != 0:
            raise ValueError(
                "dim has to be even to split on it but is {}".format(
                    t.size(dim)))
        split_point = int(len_dim / 2)
        w = t.index_select(
            dim,
            torch.tensor(
                list(range(split_point)), dtype=torch.int64,
                device=t.device)).clamp(-1., 1. - tanh_eps)

        W = t.index_select(
            dim,
            torch.tensor(
                list(range(split_point, len_dim)),
                dtype=torch.int64,
                device=t.device)).clamp(-1. + tanh_eps, 1.)
        box_val: Tensor = torch.stack((w, W), -2)

        return cls(box_val)


class TanhActivatedMinMaxBoxTensor(TanhActivatedBoxTensor):
    """
    Same as TanhActivatedBoxTensor as in it assumes input from a tanh but
    different in how it uses this input to create boxes.

    z = min((1+w)/2 , (1+W)/2)
    Z = max((1+w)/2, (1+W)/2)
    """

    @property
    def z(self) -> Tensor:
        return torch.min(self.w2z(self.data), dim=-1)

    @property
    def Z(self) -> Tensor:
        return torch.max(self.w2z(self.data), dim=-1)

    @classmethod
    def from_zZ(cls: Type[TBoxTensor], z: Tensor, Z: Tensor) -> TBoxTensor:
        raise NotImplementedError


class TanhActivatedCenterSideBoxTensor(TanhActivatedBoxTensor):
    """Same as BoxTensor but with a parameterization which is assumed to be the output
    from tanh activation. Differs from TanhActivatedBoxTensor() in the way it parameterizes the boxes

    let (*, num_dims) be the shape of output of the activations, then the BoxTensor is
    created with shape (*, zZ, num_dims/2)

    c = (w + 1)/2 => c in (0,1)

    l = (W + 1)/2 => l in (0,1)

    z = sigmoid(c - l)
    Z = sigmoid(c + l)

    """

    @classmethod
    def w2z(cls, w: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    @property
    def z(self) -> Tensor:
        c = self.data[..., 0, :]
        l = self.data[..., 1, :]

        return torch.sigmoid(c - l)

    @property
    def Z(self) -> Tensor:
        c = self.data[..., 0, :]
        l = self.data[..., 1, :]

        return torch.sigmoid(c + l)

    @classmethod
    def from_zZ(cls: Type[TBoxTensor], z: Tensor, Z: Tensor) -> TBoxTensor:
        raise NotImplementedError()
