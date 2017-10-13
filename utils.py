import numpy as np
import matplotlib.pyplot as plt
import pdb
import os
import pickle
import getcoords
from matplotlib.patches import Rectangle as rect
import cv2

def debug_plot2(frame, pts = None, roi = np.empty(0)):
    """Helper debugging plot
    
    Accepts either 4x2 as vertices or tuple of 4 as bbox params/
    """
    fig, (ax1, ax2) = plt.subplots(1,2, figsize = (12,8))
    
    if type(pts) is tuple:
        frame = draw_rectangle(frame, pts)
        pts = bbox_to_pts(pts)
        
    if not roi.size and pts is not None:
        roi = frame[pts[0][1]:pts[2][1], pts[0][0]:pts[1][0], :]
    
    if pts is not None:
        ax1.scatter(pts[:,0], pts[:,1], s = 100, c = "c",marker = "o")
        
    ax2.imshow(roi, cmap = "bone")    
    ax1.imshow(frame, cmap = "bone")
    plt.show()
    
def startdebug():
    """Start debugger here
    """
    pdb.set_trace()
    
def rectangle_to_centroid(pts):
    """Average 2D vertices to obtain rectangle centroid
    
    Paramters
    ----------
    pts : ndarray
        4 vertices defining corners of rectangle
    Returns
    ---------
    (cX, cY) : tuple
        x and y coordinates of centroid 
    """
    if isinstance(pts, tuple):
        pts = bbox_to_pts(pts)
    pts = np.int64(pts)
    # Pull out vertex coordinates
    Xs = [p[0] for p in pts]
    Ys = [p[1] for p in pts]
    # Get mean coordinates in X and Y -> centroid of bbox
    cX = np.mean(Xs, dtype = np.float32)
    cY = np.mean(Ys, dtype = np.float32)
    cZ = 0.0
    
    return np.asarray([cX, cY, cZ], dtype = np.float32)
    
def swap_coords_2d(pts):
    """Swaps x and y coordinates in n-by-2 array
    
    Utility function for compatibility with opencv axes order.
    """
    pts[:,[1,0]] = pts[:,[0,1]]
    return pts
    
def bbox_to_pts(bbox):
    """Converts tuple of 4 values to bbox vertices 
    
    """
    pts = np.array([[ bbox[0], bbox[1]], #[tl, tr, br, bl]
                    [ bbox[0]+ bbox[2], bbox[1]],
                    [ bbox[0]+ bbox[2], bbox[1]+ bbox[3]],
                    [ bbox[0], bbox[1]+ bbox[3]]], dtype = np.int32) 
    #pts = getcoords.order_points(pts)
    return pts
    
def make_filename(full_path, ext_new = "", init_fname = None):
    """Constructs filename from given path and extension
    
    If user supplies init_fname and/or new_ext, this replaces the current fname
    and ext. 
    """
    (path, ext) = os.path.splitext(full_path)
    (head, tail) = os.path.split(path)
    if ext_new: ext = ext_new;
    if init_fname:
        fname = os.path.normpath(head) + "\\" + init_fname + ext
    else:
        fname = os.path.normpath(path) + ext
    return fname
    
def save_tracking_params(video_src, save_dict, ext, init_fname = None):
    """Saves parameters to file
    
    If file already exists, it gets updated by new values by either appending,
    or overwriting them, so that uniquness prserved. 
    """
    save_name = make_filename(video_src, ext, init_fname)
    if os.path.isfile(save_name):
        saved_params = load_tracking_params(video_src, ext)
        save_dict = {**saved_params, **save_dict}
        
    if ext == ".dat":
        with open (save_name, 'wb') as outfile:
            pickle.dump(save_dict, outfile, protocol = pickle.HIGHEST_PROTOCOL)
    elif ext == ".npz":
        np.savez_compressed(fname, save_dict)
    # return some succces flag?
        
def load_tracking_params(base, ext, names = None, init_fname = None):
    """Loads parameters from file
    
    TODO: some error handling if file not available
        : can determine if file binary in advance?
    """
    load_name = make_filename(base, ext, init_fname)
    if ext == ".dat":
        with open(load_name, "rb") as infile:
            loaded = pickle.load(infile)
    elif ext == ".npz":
        loaded = np.load(fname)
        
    if names:
        dict_out = dict((k , loaded[k]) for k in names)
    else:
        dict_out = loaded
        
    return dict_out

def get_parameter_names(remove_bg, reinitialize_hsv, reinitialize_roi,
                        reinitialize_bg):
    """Prepares names of parameters to be loaded from two subsets.
    
    """
    load_bg = not reinitialize_bg
    load_hsv = not reinitialize_hsv
    load_roi = not reinitialize_roi
    
    names_init = []
    names = []
    
    if remove_bg and load_bg:
        names.append("background")
        names_init.append("background")
        
    if load_hsv:
        #names.append("roi_hist")
        names_init.append("roi_hist")
        
    if load_roi:
        names.append("pts")
        names.append("frame_pos")

    return names, names_init
    
def adjust_flags(init_flag, remove_bg, reinitialize_roi, reinitialize_hsv):
    """Updates boolean flags
    
    """
    save_init = any([reinitialize_roi, reinitialize_hsv, reinitialize_bg])
    if init_flag:
        save_init = True
        init_fname = "init"
    else:
        save_init = False
        reinitialize_bg = False
        reinitialize_hsv = False
        init_fname = None
    #return updated_flags
    
def get_in_out_names(video_src, init_flag, save_init):
    """Prepares names of files to load/save parameters from/to
    """
    data_out = video_src
    param_in = video_src
    
    if init_flag and save_init:
        init_out = make_filename(video_src, "", "init")
    else:
        init_out = None
    
    if init_flag:
        init_in = None
    else:        
        init_in = make_filename(video_src, "", "init")
        
    fnames = [data_out, param_in, init_out, init_in]
    return fnames
    
    
def draw_rectangle(frame, bbox):
    frame_vis = frame.copy().astype(np.uint8)
    tl = (int(bbox[0]), int(bbox[1])) # top-left
    br = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])) #bottom-right
    cv2.rectangle(frame_vis, tl, br, (255,0,0), thickness = 3)
    return frame_vis