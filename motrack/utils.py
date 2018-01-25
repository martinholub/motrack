import numpy as np
import matplotlib.pyplot as plt
import pdb
import os
import pickle
import getcoords
from matplotlib.patches import Rectangle as rect
import cv2
import re
import imutils
from scipy.spatial import distance as dist

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

def order_points(pts, ratio = 1):
    '''Order points of bounding box in clockwise order.
    
    Parameters
    -----------
    pts: 4x2 array of four point pairs
    ratio: scaling ratio
    
    Returns
    -----------
    array_like
        points in [top_left, top_right, bottom_right, bottom+left] order
    '''
    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]
    
    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]
    
    # Sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost

    # now that we have the top-left coordinate, use it as an
    # anchor to calculate the Euclidean distance between the
    # top-left and right-most points; by the Pythagorean
    # theorem, the point with the largest distance will be
    # our bottom-right point
    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]
    
    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    pts_out = np.array([tl, tr, br, bl], dtype="int32") * ratio
    return pts_out
    
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

def resize_frame(image, height = 860):
    '''Resize image frame.
    
    Parameters
    ------------
    image: ndaray
    height : int
        Height of output frame to be rescaled to
    
    Returns
    -----------
    image : ndarray
        resized image
    ratio : int
        scaling ratio
    '''
    ratio = image.shape[0] / height
    image = imutils.resize(image, height = height)
    return image, ratio
    
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
    try:
        os.mkdir("inits")
    except FileExistsError:
        pass
        
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
        non_decimal = re.compile(r'[^\d|\.]+')
        list_in = var_in.split()
        var_out = [int(float(non_decimal.sub("", x))) for x in list_in]
        
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

def max_width_height(box):
    """Computes maximal width and height of rotated and possibly skewed
    bounding box
    
    Parameters
    -------------
    pts : ndarray
        4x2 array of box points
        
    maxDims : tuple
        tuple of (maxWidth, maxHeight)
    """
    # Unpack the points
    (tl, tr, br, bl) = box.astype(dtype = np.float32)
    
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    return (maxWidth, maxHeight)

def apply_pad(image, pad, mode = 'symmetric'):
    """ Apply padding to an image
    
    Parameters
    ----------
    pad : tuple
        (y_pad, x_pad)
    mode : str
        mode of padding, "symmetric" by default
        
    Returns
    -------------
    image : ndarray
        Image with padding
        
    References
    --------------
    [1]  https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.pad.html
    """
    (y_pad, x_pad) = pad
    if len(image.shape) >= 2:
        pad_vector = [(y_pad, y_pad), (x_pad, x_pad)]
    elif len(image.shape) == 3:
        pad_vector.append((0, 0))
    image = np.pad(image, pad_vector, mode = mode)
    
    return image
    
def confirm_overwrite(fnames_list):
    """Confirm overwriting of files
    
    check if files exist and if they do, propmt user for overwiriting confirmation
    
    Parameters
    --------------
    fnames_list: list
        filenames to check for overwriting
    
    Returns
    --------------
    do_overwrite : bool
        boolean indicating whether to overwrite the exisitng files
    """
    is_file_list = []
    for n in fnames_list:
        is_file_list.append(os.path.isfile(n))
        
    if any(is_file_list):
        response = input("Overwrite exsiting files: [y/n]")
    
        if response == "y":
            do_overwrite = True
        elif response == "n":
            do_overwrite = False
        else:
            print("Unknown input, defaulting to `n`.")
            do_overwrite = False
    else:
        do_overwrite = True
    
    return do_overwrite

def debug_points(image, ellip, circ, pts1, pts2, pad = 0):
    """Helper function to debug transformations
    
    Plots mapping of ellipse-circle point-pair coordinates
    
    """
    import matplotlib.patches as mpatches
    import matplotlib.lines as mlines
    
    pad = 0
    imageVis = image.copy()
    (c,r) = circ
    
    majAx = ellip[1][1]
    minAx = ellip[1][0]
    angle_rad = (90 - ellip[2]) * (2*np.pi / 360)
    # Get directional shifts
    xshift = np.sin(angle_rad) * (majAx / 2)
    yshift = -np.sin(angle_rad) * (minAx / 2)
    
    fig, ax = plt.subplots(1,1, figsize = (12, 12))
    #Add circle, centre, ellipse over the image
    circle = plt.Circle((c[0], c[1]+pad), radius = r,
                        fill = False, color = "r", linewidth = 2)
    ellipse = mpatches.Ellipse((ellip[0][0], ellip[0][1] + pad),
                                np.int(minAx), np.int(majAx), angle = ellip[2],
                                fill = False, color = "b", linewidth = 4)
    ax.add_artist(circle)
    ax.add_artist(ellipse)
    
    ax.scatter( pts1[:, 0], pts1[:,1] + pad, s = 100, c = "c",
                marker = "o", label = "Circle Pts")                       
    ax.scatter( pts2[:, 0], pts2[:,1] + pad, s = 100, c = "m",
                marker = "x", label = "Ellipse Pts")
    
    linestyles = ['--', ':']
    for (ls, pts) in (zip(linestyles, [pts1, pts2])):
        majAx_line = mlines.Line2D(pts[0:2, 0], pts[0:2,1]+pad, linestyle = ls)
        minAx_line = mlines.Line2D(pts[2:4, 0], pts[2:4,1]+pad, linestyle = ls)
        ax.add_line(majAx_line)
        ax.add_line(minAx_line)
    imageVis_pad = np.pad(imageVis, ((pad, pad), (0, 0)), mode = 'symmetric')
    ax.imshow(imageVis_pad, cmap = "gray", alpha = 1)
    plt.show()
    