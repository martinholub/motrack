# test_utils_io.py
import utils
import pytest
import os
import numpy as np
import cv2

@pytest.fixture
def fname():
    return "parent\\file.ext"
@pytest.fixture
def save_dict():
    return dict(spam = "ham", eggs = 4)
 
class TestUtilsIO(object):

    def test_save_load_tracking_parameters(self, fname, save_dict):
        save_name = utils.save_tracking_params(fname, save_dict, ".dat")      
        names = list(save_dict.keys())
        load_dict = utils.load_tracking_params(fname, ".dat", names)
        os.remove(save_name)
        assert save_dict == load_dict
        
    def test_get_parameter_names(self):
        flags = [True, False, False, True]
        frame_range = [0, 1]
        names, names_init = utils.get_parameter_names(*flags)
        assert names == ["pts", "frame_pos"]
        assert names_init == ["roi_hist", "chs", "h_ranges", "pts"]

    def test_get_in_out_names(self, fname):
        fnames = utils.get_in_out_names(fname, True, True)
        assert fnames[0] == fname
        assert fnames[2] == "inits\\init.ext"
        
    def test_convert_list_string(self):
        list_in = [0, 1]
                
        str_out = utils.convert_list_str(list_in)
        list_out = utils.convert_list_str(str_out)
        assert list_in == list_out
        
        list_in_array = np.asarray(list_in)
        str_out = utils.convert_list_str(list_in_array)
        list_out = utils.convert_list_str(str_out)
        assert (np.asarray(list_out) == list_in_array).all()
        
    def test_define_video_output(self, fname):
        vid = cv2.VideoCapture("tests\\res\\test.avi")
        out_name = "vids\\test.avi"
        vid_out = utils.define_video_output(out_name, vid, 25, 5, 500)
        assert vid_out.isOpened()
        vid_out.release()
        os.remove("out_" + out_name)
    
    # def test_confirm_overwrite(self):
        # fnames = ["tests\\res\\test.avi", "tests\\res\\test.jpg"]
        # ret_val = confirm_overwrite(fnames_list)
        # assert ret_val == True