import pytest
from .modules import *
from collections import OrderedDict

@pytest.fixture(
    params = [UnitBoxes, DeltaBoxes]
)
def box_3_4_5(request):
    Boxes = request.param
    return Boxes(3,4,5)

def test_Boxes_forward_no_args(box_3_4_5):
    assert dict(box_3_4_5().shape) == {"model": 3, "box": 4, "zZ": 2, "dim": 5}

def test_Boxes_forward_list(box_3_4_5):
    unitbox_out = box_3_4_5(ntorch.tensor([1,3], names=("box",)))
    assert dict(unitbox_out.shape) == {"model": 3, "box": 2, "zZ": 2, "dim": 5}
