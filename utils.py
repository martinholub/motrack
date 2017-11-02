import numpy as np
import matplotlib.pyplot as plt
import pdb
import os
import pickle
import getcoords
from matplotlib.patches import Rectangle as rect
import cv2
import re

def debug_plot(frame, pts = None, roi = np.empty(0), cxy = (0, 0)):
    """Helper debugging plot
    
    Plot image, bbox and roi
    
    Parameters
    --------------
    frame : ndarray
        first image to plot
    pts : ndarray or tuple
        if ndarray then 4x2 defining 4 corners of rectangle, if tuple then (c,r,w,h)
        describing bounding box, will be scattered over frame [default = None]
    roi : ndarray
        second image to plot, needn't to be smaller than first one. If pts provided
        and roi not then it will be pulled out from frame [default = np.empty(0)]
    cxy : tuple
        coordinates of single point to add to plot. Useful for veryfing x,y axis.
        [default = (0, 0)]
    """
    fig, (ax1, ax2) = plt.subplots(1,2, figsize = (12,8))
    
    if type(pts) is tuple:
        frame = draw_rectangle(frame, pts)
        pts = bbox_to_pts(pts)
        
    if not roi.size and pts is not None:
        roi = frame[pts[0][1]:pts[2][1], pts[0][0]:pts[1][0], :]
    
    if pts is not None:
        ax1.scatter(pts[:,0], pts[:,1], s = 100, c = "c",marker = "o")
        
        ax1.scatter(cxy[0], cxy[1], s = 100, c = "g", marker = "x")
    
    if roi.size:   
        ax2.imshow(roi, cmap = "bone")    
    ax1.imshow(frame, cmap = "bone")
    plt.show()
    
    
def rectangle_to_centroid(pts):
    """Average 2D vertices to obtain rectangle centroid
    
    Paramters
    ----------
    pts : ndarray
        4 vertices defining corners of rectangle
    
    Returns
    ---------
    [cX, cY, cZ] : ndarray
        x, y and z coordinates of centroid.
    
    References
    ----------
    getcoords.fit_undistort
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
    
    Parameters
    ---------------
    pts : ndarray
        4 vertices defining corners of rectangle
        
    Returns
    -----------
    pts : ndarray
        same as input but with swaped first two columns
        
    References
    -------------
    [1]  https://tspp.wordpress.com/2009/10/19/x-y-coordinate-system-on-opencv/
    """
    pts[:,[1,0]] = pts[:,[0,1]]
    return pts
    
def bbox_to_pts(bbox):
    """Converts tuple of 4 values to bbox vertices
    
    Parameters
    -------------
    bbox : tuple
        (c, r ,w, h) paramaters defining bbox 

    Returns
    -------------
    pts : ndarray
        Corners of bounding box in [tl, tr, br, bl] order
        
    """
    pts = np.array([[ bbox[0], bbox[1]], #[tl, tr, br, bl]
                    [ bbox[0]+ bbox[2], bbox[1]],
                    [ bbox[0]+ bbox[2], bbox[1]+ bbox[3]],
                    [ bbox[0], bbox[1]+ bbox[3]]], dtype = np.int32) 
    #pts = getcoords.order_points(pts)
    return pts
    
def adjust_filename(fname, ext_out):
    """Replaces extension in filename.
    
    Fname can be full path, in that case only tail is returned.
        
    Parametrs
    -----------
    fname: str
    ext_out: str
    
    Returns
    -----------
    tail_out : str

    """
    (head, tail) = os.path.split(fname)
    (base, ext) = os.path.splitext(tail)
    
    tail_out = base + ext_out
    return tail_out
    
def make_filename(rel_path, ext_new = "", init_fname = None, parent = None):
    """Constructs filename from given relative path and extension
    
    If user supplies init_fname and/or new_ext, this replaces the current fname
    and ext.
    
    Parameters
    ------------
    rel_path : str
        relative path w.r.t. to cwd
    ext_new : str
        Replacement file extension, if specified must include dot [default = ""]
    init_fname : str
        Name of file that stores paramaters relevant for all processed files 
        [default = None]
    parent : str
        Name of parent folder to replace current parent in relative path. Useful
        for organizing input/output
    
    Returns
    -----------
    fname : str
        Relative path to file with filename and extension
    """
    (path, ext) = os.path.splitext(rel_path)
    (head, tail) = os.path.split(path)
    
    if parent:
        head = parent
        path = os.path.join(head, tail)
        
    if ext_new: ext = ext_new;
    
    if init_fname:
        fname = os.path.normpath(head) + "\\" + init_fname + ext
    else:
        fname = os.path.normpath(path) + ext
        
    return fname
    
def save_tracking_params(video_src, save_dict, ext, init_fname = None):
    """Saves parameters to file
    
    If file already exists, it gets updated by new values by either appending,
    or overwriting them, so that uniquness preserved.
    
    Parameters
    ------------
    video_src : str
        path to source video file
    save_dict : dict
        dictionary with key,value pairs to save to file
    ext : str
        Filetype to use for saving, either ".npz" or ".dat" [default = ".dat"]
    init_fname : str
        Name of file that stores paramaters relevant for all processed files 
    """
    save_name = make_filename(video_src, ext, init_fname, parent = "inits")
    if os.path.isfile(save_name):
        saved_params = load_tracking_params(video_src, ext)
        save_dict = {**saved_params, **save_dict}
        
    if ext == ".dat":
        with open (save_name, 'wb') as outfile:
            pickle.dump(save_dict, outfile, protocol = pickle.HIGHEST_PROTOCOL)
    elif ext == ".npz":
        np.savez_compressed(fname, save_dict)
    
    return save_name # return some succces flag?
        
def load_tracking_params(base, ext, names = None, init_fname = None):
    """Loads parameters from file
    
    If file doesn't exist, returns empty dictionary
    
    Parameters
    --------------
    base : str
        relative path to file
    ext : str
        file extension, must be either ".npz" or ".dat" [default = ".dat"]
    names : list
        names of variables to load from file
    
    Returns
    ------------
    dict_out : dict
        loaded variables as key,value pairs
    """
    load_name = make_filename(base, ext, init_fname, parent = "inits")
    if not os.path.isfile(load_name):
        return {}
        
    if ext == ".dat":
        with open(load_name, "rb") as infile:
            loaded = pickle.load(infile)
    elif ext == ".npz":
        loaded = np.load(fname)
    
    try:
        if names:
            dict_out = dict((k , loaded[k]) for k in names)
        else:
            dict_out = loaded
    except:
        return {}
        
    return dict_out

def get_parameter_names(remove_bg, reinitialize_hsv, reinitialize_roi,
                        reinitialize_bg):
    """Prepares names of parameters to be loaded from two subsets
    
    Parameters
    ----------------
    diverse flags : bool
        flags indicating what setups are reinitialized in current run
    
    Returns
    ------------------
    names : list
        names of file-specific variables that should be saved /loaded 
    names_init : list
        names of variables that should be saved / loaded and that are constant
        across all files in directory
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
        names_init.append("chs")
        names_init.append("h_ranges")
        
    if load_roi:
        names.append("pts")
        names_init.append("pts")
        names.append("frame_pos")

    return names, names_init
    
def get_in_out_names(video_src, init_flag, save_init):
    """Prepares names of files to load/save parameters from/to
    
    Parameters
    --------------
    init_flag : bool
        flag indicating if the current run is initializing
    save_init : bool
        flag indicating if we want to save some parameters to a file that is accessable
        during processing of all files
    
    Returns
    ---------------
    fnames : list
        List of [data_out, param_in, init_out, init_in] relative paths to files,
        with video-specific extension preserved
    """
    data_out = video_src
    param_in = video_src
    
    if init_flag and save_init:
        init_out = make_filename(video_src, "", "init", parent = "inits")
    else:
        init_out = None
    
    if init_flag:
        init_in = None
    else:        
        init_in = make_filename(video_src, "", "init", parent = "inits")
        
    fnames = [data_out, param_in, init_out, init_in]
    return fnames
    
    
def draw_rectangle(frame, bbox):
    """Draw a bounding box in a frame
    
    Returns
    -----------
    frame_vis : ndarray
        frame with bbox drawn
    """
    frame_vis = frame.copy().astype(np.uint8)
    tl = (int(bbox[0]), int(bbox[1])) # top-left
    br = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])) #bottom-right
    cv2.rectangle(frame_vis, tl, br, (255,0,0), thickness = 3)
    return frame_vis
    
def convert_list_str(var_in):
    """Converts between list/array and str and vice versa

    Useful when passing command line arguments to subprocess.
    """
    if type(var_in) is np.ndarray:
        var_in = var_in.tolist()
    if type(var_in) is list:
        var_out = str(var_in)
        
    if type(var_in) is str:
        non_decimal = re.compile(r'[^\d]+')
        list_in = var_in.split()
        var_out = [int(non_decimal.sub("", x)) for x in list_in]
        
    return var_out
        
def define_video_output(video_src, vid, fps, step, out_height):
    """Creates object for storing video frames
    
    Videos are stored in folder "out_vids" and their filenames are prepended with 
    "out_". The folder is created if necessary.
    
    Parameters
    ----------
    video_src : str
        path to video being recorded
    vid : cv.VideoCapture object
        video being recorded
    fps : float
        frames per second of former video
    step : float
        Frame-span used for running average
    out_height : int
        height of video to be output
    
    Returns
    -------------
    vid_out : cv2.VideoWriter object
        video to write frames to
        
    References
    ------------
    [1]  https://www.pyimagesearch.com/2016/02/22/writing-to-video-with-opencv/            
    """
    vid_name = make_filename("out_" + video_src, ".avi")
    
    try:
        os.mkdir("out_vids")
    except FileExistsError:
        pass
        
    width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
    height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT) # float
    # Define the codec and create VideoWriter object
    ratio = out_height / height
    width = np.int(ratio * width)
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    vid_out = cv2.VideoWriter(vid_name,fourcc, fps / step, (width, out_height))
    #vid_out = cv2.VideoWriter(vid_name,fourcc, 5, (675, 500))
    return vid_out