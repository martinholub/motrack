# test_motrack_process.py
import numpy as np
import motrack
import cv2
import numpy as np
import pytest

@pytest.fixture
def image():
    return cv2.imread("tests\\res\\test.jpg")
@pytest.fixture
def roi_hist_tuple(image):
    vid = "path\\to\\video\\file.ext"
    return motrack.get_roi_hist(image, vid)
@pytest.fixture
def bg_tuple(image):
    bbox = (50, 75, 125, 75)
    #(x, y, dx, dy), where x horizontal, y vertical with origin at top-left
    return motrack.segment_background(image, bbox)
@pytest.fixture
def params():
    return dict(kSize_gauss = (1, 1), sigmaX = 1, kSize_canny = (1, 1), 
                padding = 10)
    
class TestMotrackProcess(object):
        
    def test_average_frames(self):
        vid = cv2.VideoCapture("tests\\res\\test.avi")
        retval, frame = vid.read()
        frame_avg = np.zeros_like(frame, dtype = np.float32)
        frame_avg, _, _ = motrack.average_frames(vid, frame_avg)
        assert np.round(np.sum(frame_avg) / frame_avg.size) == 92.0
        vid.release()
        
    def test_get_roi_hist(self, roi_hist_tuple):
        roi_hist = roi_hist_tuple[0]
        assert np.round(np.sum(roi_hist)) == 6831.0
    
    def test_prob_mask_hsv(self, roi_hist_tuple, image):
        prob_mask = motrack.prob_mask_hsv(image, *roi_hist_tuple)
        assert np.sum(prob_mask) == 1757037
    
    def test_segment_background(self, bg_tuple):
        _, mask_fgd = bg_tuple
        assert np.sum(mask_fgd) == 7920
    
    def test_subtract_background(self, bg_tuple, image):
        bg, _ = bg_tuple
        bg_removed = motrack.subtract_background(image, bg)
        assert np.round(np.sum(bg_removed) / bg_removed.size) == 17
        
    def test_label_contours_type2(self, image, params):
        (ret_img, cnts) = motrack.label_contours(image, **params)
        assert sum([len(c) for c in cnts]) == 434
    
    def test_label_contours_type1(self, image, params):
        (ret_img, cnts) = motrack.label_contours(image, type = 1, **params)
        assert sum([len(c) for c in cnts]) + len(cnts) == 27
        
    def test_update_bbox_location(self, image, params):
        bbox = (50, 75, 125, 75)
        bbox_new, bbox_new_pad = motrack.update_bbox_location(image, bbox, **params)
        assert bbox_new == (40, 75, 71, 77)
        assert bbox_new_pad == (30, 65, 91, 97)
        
        