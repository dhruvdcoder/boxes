from boxes.box_wrapper import *
import torch
import logging
import numpy as np
import pytest

logger = logging.getLogger(__name__)


def test_simple_creation():
    tensor = torch.tensor(np.random.rand(3, 2, 3))
    box_tensor = BoxTensor(tensor)
    assert (tensor.data.numpy() == box_tensor.data.numpy()).all()
    assert isinstance(box_tensor, BoxTensor)
    tensor = torch.tensor(np.random.rand(2, 10))
    box_tensor = BoxTensor(tensor)
    assert (tensor.data.numpy() == box_tensor.data.numpy()).all()
    assert isinstance(box_tensor, BoxTensor)
    tensor = torch.tensor(np.random.rand(10, 2, 3, 2, 10))
    box_tensor = SigmoidBoxTensor(tensor)
    assert (tensor.data.numpy() == box_tensor.data.numpy()).all()
    assert isinstance(box_tensor, BoxTensor)
    assert isinstance(box_tensor, SigmoidBoxTensor)


def test_no_copy_during_creation():
    tensor = torch.tensor(np.random.rand(3, 2, 3))
    box_tensor = BoxTensor(tensor)
    tensor[0][0][0] = 1.
    assert (tensor.data.numpy() == box_tensor.data.numpy()).all()

    tensor = torch.tensor(np.random.rand(3, 2, 3))
    box_tensor = BoxTensor(tensor)
    box_tensor.data[0][0][0] = 1.
    assert (tensor.data.numpy() == box_tensor.data.numpy()).all()


def test_validation_during_creation():
    tensor = torch.tensor(np.random.rand(3))
    with pytest.raises(ValueError):
        box_tensor = BoxTensor(tensor)
    tensor = torch.tensor(np.random.rand(3, 11))
    with pytest.raises(ValueError):
        box_tensor = BoxTensor(tensor)
    tensor = torch.tensor(np.random.rand(3, 3, 3))
    with pytest.raises(ValueError):
        box_tensor = BoxTensor(tensor)


def test_creation_from_zZ():
    shape = (3, 1, 5)
    z = torch.tensor(np.random.rand(*shape))
    Z = torch.tensor(np.random.rand(*shape))
    box = BoxTensor.from_zZ(z, Z)
    assert box.data.shape == (3, 1, 2, 5)


def test_creation_from_zZ_sigmoid_box():
    z1 = (torch.tensor([[1, 1], [1, 1]]).float()) / 6.
    Z1 = (torch.tensor([[3, 5], [2, 3]]).float()) / 6.
    box1 = SigmoidBoxTensor.from_zZ(z1, Z1)
    z2 = (torch.tensor([[2, 0], [3, 2]]).float()) / 6.
    Z2 = (torch.tensor([[6, 2], [4, 4]]).float()) / 6.
    box2 = SigmoidBoxTensor.from_zZ(z2, Z2)

    assert np.allclose(z1.data.numpy(), box1.z.data.numpy())
    assert np.allclose(Z1.data.numpy(), box1.Z.data.numpy())
    assert np.allclose(z2.data.numpy(), box2.z.data.numpy())
    assert np.allclose(Z2.data.numpy(), box2.Z.data.numpy())


def test_copying_during_creation_from_zZ():
    shape = (3, 1, 5)
    z = torch.tensor(np.random.rand(*shape))
    Z = torch.tensor(np.random.rand(*shape))
    box = BoxTensor.from_zZ(z, Z)
    z[0][0][0] = 1.
    assert not np.allclose(box.data.numpy()[0][0][0][0],
                           (z).data.numpy()[0][0][0])


def test_validation_during_creation_from_zZ():
    shape_z = (2, 5)
    shape_Z = (3, 5)
    z = torch.tensor(np.random.rand(*shape_z))
    Z = torch.tensor(np.random.rand(*shape_Z))
    with pytest.raises(ValueError):
        box = BoxTensor.from_zZ(z, Z)


def test_intersection():
    box1 = BoxTensor(torch.tensor([[[1, 1], [3, 5]], [[1, 1], [2, 3]]]))
    box2 = BoxTensor(torch.tensor([[[2, 0], [6, 2]], [[3, 2], [4, 4]]]))
    res = BoxTensor(torch.tensor([[[2, 1], [3, 2]], [[3, 2], [2, 3]]]))
    assert (res.data.numpy() == box1.intersection(box2).data.numpy()).all()


def test_intersection_sigmoid_box():
    z1 = (torch.tensor([[1, 1], [1, 1]]).float()) / 6.
    Z1 = (torch.tensor([[3, 5], [2, 3]]).float()) / 6.
    box1 = SigmoidBoxTensor.from_zZ(z1, Z1)
    z2 = (torch.tensor([[2, 0], [3, 2]]).float()) / 6.
    Z2 = (torch.tensor([[6, 2], [4, 4]]).float()) / 6.
    box2 = SigmoidBoxTensor.from_zZ(z2, Z2)
    resz = torch.tensor([[2, 1], [3, 2]]).float() / 6.
    resZ = torch.tensor([[3, 2], [2, 3]]).float() / 6.
    res = SigmoidBoxTensor.from_zZ(resz, resZ)
    intersection = box1.intersection(box2)
    assert np.allclose(res.data.numpy(), intersection.data.numpy())
    assert np.allclose(res.z.data.numpy(), intersection.z.data.numpy())
    assert np.allclose(res.Z.data.numpy(), intersection.Z.data.numpy())


def test_join():
    box1 = BoxTensor(torch.tensor([[[1, 1], [3, 5]], [[1, 1], [2, 3]]]))
    box2 = BoxTensor(torch.tensor([[[2, 0], [6, 2]], [[3, 2], [4, 4]]]))
    res = BoxTensor(torch.tensor([[[1, 0], [6, 5]], [[1, 1], [4, 4]]]))
    assert (res.data.numpy() == box1.join(box2).data.numpy()).all()


def test_clamp_vol():
    box1 = BoxTensor(torch.tensor([[[1, 1], [3, 5]], [[1, 1], [2, 3]]]))
    vol = torch.tensor([8, 2])
    assert (box1.clamp_volume().data.numpy() == vol.data.numpy()).all()
    # with flipped box
    box1 = BoxTensor(torch.tensor([[[1, 1], [3, 5]], [[2, 3], [1, 1]]]))
    vol = torch.tensor([8, 0])
    assert (box1.clamp_volume().data.numpy() == vol.data.numpy()).all()


def test_log_clamp_vol():
    box1 = BoxTensor(
        torch.tensor([[[1, 1], [3, 5]], [[1, 1], [2,
                                                  3]]]).to(dtype=torch.float))
    vol = torch.log(torch.tensor([8, 2]).to(dtype=torch.float))
    assert np.allclose(box1.log_clamp_volume().data.numpy(), vol.data.numpy())
    # with flipped box
    box1 = BoxTensor(
        torch.tensor([[[1, 1], [3, 5]], [[2, 3], [1,
                                                  1]]]).to(dtype=torch.float))
    vol = torch.log(torch.tensor([8, 0 + torch.finfo(torch.float).tiny]))
    vol[1] = vol[1] * (2
                       )  # need this because in sum(log()) eps is added twice,
    # once of each dim because both are flipped here
    assert np.allclose(box1.log_clamp_volume().data.numpy(), vol.data.numpy())


def test_simple_grad():
    class Identity(torch.nn.Module):
        def forward(self, inp):
            return SigmoidBoxTensor(inp).data

    layer = Identity()
    batch_size = 5
    seq_len = 3
    inp_dim = 10
    inp_shape = (batch_size, seq_len, inp_dim)

    def test_case():
        inp = torch.tensor(
            np.random.rand(*inp_shape), requires_grad=True).double()
        torch.autograd.gradcheck(layer, inp)


def test_from_zZ_grad():
    class Identity(torch.nn.Module):
        def forward(self, inp):
            z, Z = inp
            return SigmoidBoxTensor.from_zZ(z, Z).data

    layer = Identity()
    batch_size = 5
    seq_len = 3
    inp_dim = 10
    inp_shape = (batch_size, seq_len, inp_dim)

    def test_case():
        z = torch.tensor(
            np.random.rand(*inp_shape), requires_grad=True).double()
        Z = torch.tensor(
            np.random.rand(*inp_shape), requires_grad=True).double()

        torch.autograd.gradcheck(layer, (z, Z))


def test_from_split_grad():
    class Identity(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.p = torch.nn.Parameter(torch.tensor(.2))

        def forward(self, inp):
            inp = inp * self.p
            res = SigmoidBoxTensor.from_split(inp, -1).data
            return res

    layer = Identity()
    batch_size = 5
    seq_len = 3
    inp_dim = 10
    inp_shape = (batch_size, seq_len, inp_dim)

    def test_case():
        inp = torch.tensor(
            np.random.rand(*inp_shape), requires_grad=True).double()
        torch.autograd.gradcheck(layer, inp)


if __name__ == '__main__':
    test_creation_from_zZ_sigmoid_box()