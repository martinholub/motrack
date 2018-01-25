# test_motrack_output.py
import pytest
import numpy as np
import motrack
import os

@pytest.fixture
def dists():
    dists = [[0] * 10, [1] * 20, [300] * 1, [0] * 29]
    dists = [j for i in dists for j in i]
    return dists
    
@pytest.fixture
def times():
    return np.arange(0, 60)
    
@pytest.fixture
def num_nonzeros():
    array1 = np.ones(30, dtype = np.int32) * 10
    # array2 = np.random.randint(40001, 60000, 30, dtype = np.int32)
    array2 = np.ones(30, dtype = np.int32) * 50000
    num_nonzeros = [array1, array2]
    num_nonzeros = [j for i in num_nonzeros for j in i]
    return num_nonzeros   
 
class TestMotrackOutput(object):

    def test_flag_file_review(self, dists, num_nonzeros,  times):
        fname = "tests\\res\\out_test.txt"
        with open(fname, "w") as f:
            f = motrack.flag_file_review(f, dists, num_nonzeros, times)
            curr_pos = f.tell()
            assert curr_pos == 52
        os.remove(fname)
        
    def test_compute_distance(self):
        c0 = np.asarray([100, 100, 0], dtype = np.float32)
        c1 = np.asarray([200, 200, 0], dtype = np.float32)
        centroids = [c0, c1]
        i = 1
        times = [0, 1]
        M = np.load("src\\projection_matrix.npy")
        scale = np.load("src\\scaling_factor.npy")
        dist, _, _, _ = motrack.compute_distance(centroids, c1, i, M, scale)
        assert np.round(dist) == 85.0 # 81 was for old diameter
        
        
        