#test_process.py
import utils
import numpy as np
import pytest

@pytest.fixture
def pts():
    return np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype = np.int32)
@pytest.fixture
def bbox():
    return (0, 0, 10, 10)

class TestUtilsCoords(object):    
    def test_rectangle_to_centroid(self, pts):
        res = utils.rectangle_to_centroid(pts)
        ver = np.asarray([5, 5, 0], dtype = np.float32)
        assert (res == ver).all() == True
        
    def test_swap_coords_2d(self, pts):
        swap1 = utils.swap_coords_2d(pts)
        swap2 = utils.swap_coords_2d(pts)
        assert (swap2 == pts).all()
        
    def test_bbox_to_pts(self, bbox, pts):
        res = utils.bbox_to_pts(bbox)
        assert (res == pts).all()