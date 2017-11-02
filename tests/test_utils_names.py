# test_utils_names.py
import utils
import pytest

@pytest.fixture
def fname():
    return "parent\\file.ext"

class TestUtilsNames(object):

    def test_adjust_filename(self):
        assert utils.adjust_filename( "/path/to/some/file/file.ext1", ".ext2") \
                                        == "file.ext2"
    def test_make_filename(self, fname):
        res = utils.make_filename(fname, ".new_ext", parent = "new_parent")
        assert res == "new_parent\\file.new_ext"
    
    def test_make_filename_init(self, fname):
        res = utils.make_filename(fname, init_fname = "init")
        assert res == "parent\\init.ext"
    
    