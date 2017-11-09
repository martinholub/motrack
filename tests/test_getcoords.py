# test_motrack_output.py
import pytest
import numpy as np
import getcoords
import os
import cv2
import subprocess

@pytest.fixture
def image():
    return cv2.imread("tests\\res\\test2.png")
@pytest.fixture
def contour_tuple(image):
    cnt, gray = getcoords.find_contours(image)
    return cnt, gray
@pytest.fixture
def ellip(contour_tuple):
    cnt, gray = contour_tuple
    ellip, _, _ = getcoords.fit_ellipse(cnt, gray)
    return ellip
@pytest.fixture
def mask_tuple(contour_tuple, ellip):
    pts, mask, _ = getcoords.mask_box_ellip(contour_tuple[1], ellip)
    return pts, mask
@pytest.fixture
def circle_tuple():
    num_pix = 220
    mask = np.zeros((num_pix, num_pix), dtype = np.uint8)
    cxy = (np.int(num_pix/2), np.int(num_pix/2))
    radius = np.int(num_pix/5)
    cv2.circle(mask, cxy, radius , color = (255, 0, 0), thickness = -1)
    return mask, radius

    
class TestGetcoords(object):
    
    def test_find_contours(self, contour_tuple):
        cnt, _ = contour_tuple
        assert len(cnt) == 386
        
    def test_fit_ellipse(self, ellip):
        assert np.round(sum([i for j in ellip[:2] for i in j]) + ellip[2]) == 3627.0
    
    def test_mask_box_ellip(self, mask_tuple):
        pts, mask = mask_tuple
        assert np.sum(pts) == 12492
        
    def test_projective_transform(self, contour_tuple, mask_tuple):
        _, gray = contour_tuple
        _, mask = mask_tuple
        M, scale, _, warped_mask = getcoords.projective_transform(gray, mask, D = 1)
        assert np.round(np.sum(M), 2) == 1184.49
        assert np.round(scale, 3) == 0.006
    
    def test_find_scale(self, circle_tuple):
        mask, radius = circle_tuple
        scale = getcoords.find_scale(mask, D = 2*radius)
        assert abs(1 - scale) < 0.1
        
    