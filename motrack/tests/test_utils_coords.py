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
@pytest.fixture
def frame():
    return np.ones((20, 20), dtype = np.float32) * 5

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
        
    def test_resize_frames(self, frame):
        nrows, ncols = frame.shape[:2]
        frame_out1, ratio1 = utils.resize_frame(frame)
        frame_out2, ratio2 = utils.resize_frame(frame_out1, nrows)
        assert np.round(ratio1*ratio2, 3) == 1.0
        assert frame_out2.shape == frame.shape
        
    def test_order_points(self, pts):
        pts_new = utils.order_points(np.random.permutation(pts))
        assert (pts_new == pts).all()
        
    def test_max_width_height(self, pts):
        (maxWidth, maxHeight) = utils.max_width_height(pts)
        assert maxWidth + maxHeight == 20
    
    def test_apply_pad(self, frame):
        pad = 10
        frame_out = utils.apply_pad(frame, (pad, pad))
        assert sum(frame.shape) + 4*pad == sum(frame_out.shape)
        