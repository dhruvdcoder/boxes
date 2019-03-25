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
    assert box_3_4_5().shape == (3,4,2,5)

def test_Boxes_forward_list(box_3_4_5):
    unitbox_out = box_3_4_5([1,3])
    assert unitbox_out.shape == (3,2,2,5)
