import pytest
from hypothesis import given, example
from hypothesis import strategies as st
from .box_operations import *
from collections import defaultdict
from numpy import random

test_cases = list()
test_cases.append(defaultdict(lambda: None, dict(
    boxes=torch.tensor([
            [[0.1, 0.2], [0.9, 0.5]],
            [[0.3, 0.1], [0.8, 1.0]],
        ])[None],
    # intersection = (torch.tensor([0.3, 0.8])[None, [0.8, 0.5]])[None, None],
    volumes=torch.tensor([0.24, 0.45])[None, :],
    min_vol=0.3,
    vol_func=clamp_volume,
    small_boxes=torch.tensor([1, 0], dtype=torch.uint8)[None, :],
)))


def skip_unless_parameters_exist(**kwargs):
    missing_vars = []
    for name, value in kwargs.items():
        if value is None:
            missing_vars.append(name)
    if len(missing_vars) > 0:
        reason = "Not enough info to complete this test case.\n"
        reason += "To check this case, set the following variables:\n"
        reason += ", ".join(missing_vars)
        pytest.skip(reason)
        return False
    else:
        return True


@pytest.mark.parametrize("boxes, vol_func, volumes",
     [(t["boxes"], t["vol_func"], t["volumes"]) for t in test_cases])
def test_volume(boxes, vol_func, volumes):
    skip_unless_parameters_exist(vol_func=vol_func, volumes=volumes)
    Z = boxes[:, :, 1]
    z = boxes[:, :, 0]
    out = vol_func(Z - z)
    assert (out == volumes).all()


@pytest.mark.parametrize("boxes, int_boxes",
     [(t["boxes"], t["int_boxes"]) for t in test_cases])
def test_intersection_boxes(boxes, int_boxes):
    skip_unless_parameters_exist(int_boxes=int_boxes)
    Z = boxes[:, :, 1]
    z = boxes[:, :, 0]
    out = intersection(z, Z)
    assert (out == int_boxes).all()


@pytest.mark.parametrize("boxes, vol_func, min_vol, small_boxes",
                         [(t["boxes"], t["vol_func"], t["min_vol"], t["small_boxes"]) for t in test_cases])
def test_detect_small_boxes(boxes, vol_func, min_vol, small_boxes):
    skip_unless_parameters_exist(vol_func=vol_func, min_vol=min_vol, small_boxes=small_boxes)
    out = detect_small_boxes(boxes, vol_func, min_vol)
    assert (out == small_boxes).all()


@pytest.mark.parametrize("boxes, boxes_ind, min_vol, vol_func",
                         [(t["boxes"], t["small_boxes"], t["min_vol"], t["vol_func"]) for t in test_cases])
def test_replace_Z_by_cube(boxes, boxes_ind, min_vol, vol_func):
    out = replace_Z_by_cube(boxes, boxes_ind, min_vol)
    assert (vol_func(out - boxes[:,:,0][boxes_ind]) == min_vol).all()


@pytest.mark.parametrize("boxes, boxes_ind, min_vol, vol_func",
                         [(t["boxes"], t["small_boxes"], t["min_vol"], t["vol_func"]) for t in test_cases])
def test_replace_Z_by_cube_inplace(boxes, boxes_ind, min_vol, vol_func):
    replace_Z_by_cube_(boxes, boxes_ind, min_vol)
    assert (vol_func(boxes[:,:,1] - boxes[:,:,0]) >= min_vol).all()
    assert detect_small_boxes(boxes, vol_func, min_vol).sum() == 0


@pytest.fixture(params=[()])
def unitboxes(request):
    boxes_in = request.param
    num_models, num_boxes, _, dim = boxes_in.shape
    boxes = UnitBoxes(num_models, num_boxes, dim)
    return boxes


@given(num_models=st.integers(1,10), num_boxes=st.integers(1,1000), dim=st.integers(1,100), min_vol=st.floats(1e-6,1e-1))
def test_unitboxes_replace_Z_by_cube(num_models, num_boxes, dim, min_vol):
    unitboxes = UnitBoxes(num_models, num_boxes, dim)
    small_boxes = detect_small_boxes(unitboxes.boxes, clamp_volume, min_vol)
    Z = replace_Z_by_cube(unitboxes.boxes, small_boxes, min_vol)
    assert (clamp_volume(Z - unitboxes.boxes[:,:,0][small_boxes]) >= min_vol-1e-6).all()


@given(num_models=st.integers(1,10), num_boxes=st.integers(1,1000), dim=st.integers(1,100), min_vol=st.floats(1e-6,1e-1))
def test_unitboxes_replace_Z_by_cube_inplace(num_models, num_boxes, dim, min_vol):
    unitboxes = UnitBoxes(num_models, num_boxes, dim)
    small_boxes = detect_small_boxes(unitboxes.boxes, clamp_volume, min_vol)
    replace_Z_by_cube_(unitboxes.boxes, small_boxes, min_vol)
    assert (detect_small_boxes(unitboxes.boxes, clamp_volume, min_vol - 1e-6) == 0).all()
