'''
Mouse Motion Tracker
====================

TBA

Usage
-----
motrack.py -video "path/to/file.ext"

Keys
----
ESC - exit
'''
#!/usr/bin/env python

import numpy as np
import cv2
import argparse
import pdb
import getcoords
import params as p
import sys
import os
from old_stuff import debug_plot2
from preprocess import find_frame_video
import pickle

# Parse arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type = str, help= "path to the video file")
ap.add_argument("-i", "--init_fname", type = str, help= "filename for initialization")
#ap.add_argument("-nr", "--new-roi", type = bool, help= "Select new roi")
#ap.add_argument("-nh", "--new-hist", type = bool, help = "Select new hsv range")

def startdebug():
    """Start debugger here
    """
    pdb.set_trace()
    
def from_preset(video_src):
    """Use pre-set values to speed up processing.
    
    References
    -------------
    getcoords.select_roi_video
    """
    frame_pos = 650.0
    pts = np.array([[ 520, 1259], #[tl, tr, br, bl]
                    [ 626, 1259],
                    [ 626, 1434],
                    [ 520, 1434]], dtype = np.int32)
    (vid, frame) = getcoords.go_to_frame([], frame_pos, video_src,
                                return_frame = True)
    roi = frame[pts[0][0]:pts[1][0], pts[0][1]:pts[2][1],:]
    
    return pts, roi, vid, frame_pos, frame
    
def get_roi_hist(   roi_rgb, vid, background = np.empty(0), frame_pos = [], 
                    reinitialize = False):
    """Converts ROI to histogram
    
    Parameters
    ----------
    roi_rgb : ndarray
        3-channel RGB image of region of interest
    vid : cv2.VideoCapture object or str
        If str then vid is path to the video file
    frame_pos : float
        number of frame corresponding to current position in video [defeault = []]
    
    Returns
    ------------
    roi_hist : ndarray
        n-dimensional histogram
    h_sizes : list
        number of bins in each dimension 
    h_ranges : list
        limits in corresponding dimensions, e.g. [minx, maxx, miny, maxy]
    chs : list
        chnannels to compute histogram on, corresponds to n
    
    References:
    ..[1] http://docs.opencv.org/3.3.0/dd/d0d/tutorial_py_2d_histogram.html
    """
    video_src = []
    if type(vid) is str:
        video_source = vid
        vid = []
        
    chs = p.chs
    h_sizes = p.h_sizes
    h_ranges = p.h_ranges
    
    (hsv_lowerb, hsv_upperb) = select_hsv_range(vid, video_src, background,
                                                frame_pos, reinitialize)
    
    roi_hsv = cv2.cvtColor(roi_rgb, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(roi_hsv, hsv_lowerb , hsv_upperb)
    roi_hist = cv2.calcHist(images = [roi_hsv], channels = chs, mask = mask,
                            histSize = h_sizes, ranges = h_ranges)                       
    # Normalize in-place between alpha and beta
    cv2.normalize(  src = roi_hist, dst = roi_hist,  alpha = 0, beta = 255,
                    norm_type = cv2.NORM_MINMAX)
    return roi_hist

def select_hsv_range(   vid, video_source, background = np.empty(0), 
                        frame_pos = [], reinitialize = False):
    """Select object hsv range
    
    Interactively selects HSV ranges that threshold tracked object against background
    
    Parameters
    ----------
    vid : cv2.VideoCapture object
        Already opened video object. If empty, video from video_source is read. 
    video_source : str
        path to video file
    frame_pos : float
        number of frame corresponding to current position in video [default = []]
    reinitialize : bool
        whether to look for new HSV limits or take preset [default = False]
    
    Returns
    ----------
    hsv_lowerb : array_like
        vector of lower HSV limits [hlow, slow, vlow]
    hsv_upperb : array_like
        vector of upper HSV limits [hhigh, shigh, vhigh]
    
    References
    ------------
    ..[1] https://botforge.wordpress.com/2016/07/02/basic-color-tracker-using-opencv-python/
    ..[2] http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_trackbar/py_trackbar.html
    """
    if reinitialize:
        (_, frame) = getcoords.go_to_frame( vid, frame_pos, video_source, 
                                            return_frame =True)
        frame_avg = np.zeros_like(frame, dtype = np.float32)
        
        frame, _, _ = average_frames(vid, frame_avg)
        if background.size: #if exists
            frame = subtract_background(frame, background)
                    
        frame, _ = getcoords.resize_frame(frame, height = p.height_resize)
                
        hh='Hue High'
        hl='Hue Low'
        sh='Saturation High'
        sl='Saturation Low'
        vh='Value High'
        vl='Value Low'
        cv2.namedWindow("Select HSV Range")
        cv2.resizeWindow('Select HSV Range', frame.shape[1], frame.shape[0] )
        print("Change ranges on sliders and press ENTER to update")
        
        def nothing(x):
            pass

        cv2.createTrackbar(hl, 'Select HSV Range',0,179, nothing) # ~180 deg
        cv2.createTrackbar(hh, 'Select HSV Range',0,179, nothing)
        cv2.createTrackbar(sl, 'Select HSV Range',0,255, nothing)
        cv2.createTrackbar(sh, 'Select HSV Range',0,255, nothing)
        cv2.createTrackbar(vl, 'Select HSV Range',0,255, nothing)
        cv2.createTrackbar(vh, 'Select HSV Range',0,255, nothing)

        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        try:
            while(1):
                #read trackbar positions for all
                HL=cv2.getTrackbarPos(hl, 'Select HSV Range')
                HH=cv2.getTrackbarPos(hh, 'Select HSV Range')
                SL=cv2.getTrackbarPos(sl, 'Select HSV Range')
                SH=cv2.getTrackbarPos(sh, 'Select HSV Range')
                VL=cv2.getTrackbarPos(vl, 'Select HSV Range')
                VH=cv2.getTrackbarPos(vh, 'Select HSV Range')
                #make array for final values
                hsv_lowerb=np.array([HL, SL, VL])
                hsv_upperb=np.array([HH, SH, VH])
                
                #apply the range on a mask
                mask = cv2.inRange(frame_hsv, hsv_lowerb, hsv_upperb)
                res = cv2.bitwise_and(frame_hsv, frame_hsv, mask = mask)
                
                cv2.imshow('Select HSV Range', res)
                k = cv2.waitKey(0) & 0xFF
                if k == 27:
                    break
        finally:
            cv2.destroyAllWindows()
    else: # apply preset
        if background.size: #if exists
            hsv_lowerb = np.array([0, 0, 100])
            hsv_upperb = np.array([179,200,255])
        else:
            hsv_lowerb = np.array([0, 50, 0]) # without bg substraction
            hsv_upperb = np.array([179, 160, 100]) # without bg substraction
        
    return (hsv_lowerb, hsv_upperb)
    
def prob_mask_hsv(frame_avg, roi_hist, h_ranges, chs, **kwargs):
    """Compute probability mask based on roi histogram
    
    Look for best match with a historgam in the image.
    
    Parameters
    -----------
    frame_avg : ndarray
        3channel RGB image of the whole frame, optinoally averaged over several
        frames
    roi_hist : ndarray
        n-dimensional histogram
    h_ranges : list
        limits in corresponding dimensions, e.g. [minx, maxx, miny, maxy]
    chs : list
        chnannels to compute histogram on, corresponds to n
    
    Returns
    ------------
    prob_mask : ndarray
        2D probability mask of same size as frame_avg[:,:,0]
    frame_binary : ndarray
        2D mask of detected object
    """
    frame_avg_hsv = cv2.cvtColor(frame_avg, cv2.COLOR_BGR2HSV)
    prob_mask = cv2.calcBackProject(images = [frame_avg_hsv], channels = chs, 
                                    hist = roi_hist, ranges =h_ranges, scale= 1)
    prob_mask, frame_binary = check_min_area(   prob_mask, frame_avg, **kwargs)
            
    return prob_mask, frame_binary

def output_data(centroids, times, video_src):
    """Writes select data into text file
    
    Saves centroid coordinates, traveled distance and elapsed time to file. 
    output_data is called in postprocessing. 
    
    Parameters
    -----------
    centroids : list
        list of centroids of tracked object sampled at times
    times : list
        List of sampling times. Sampling every step-th frame.
        
    References
    -----------
    getcoords.find_scale, getcoords.projective_transform
           
    """
    # Initalize variables that hold sums
    total_dist = 0
    
    scale = np.load("scaling_factor.npy")
    M = np.load("projection_matrix.npy")
    
    fname_out = make_filename(video_src, ".txt")
    
    with open(fname_out, "w") as f:
        f.write("No.,cX,cY,time,dist\n")
        for i, cent in enumerate(centroids):
            #Skip point if we have lost track
            if  np.any(cent[:2] == 0) or (i > 0 and np.any(centroids[i-1][:2] == 0)):
                # f.write("#Discarding line! {:0.2f},{:0.2f} -> {:0.2f},{:0.2f}\n".format(centroids[i-1][0], centroids[i-1][1], cent[0], cent[1]))
                continue;
            
            cent_warped = np.dot(cent, M)
            cX = cent_warped[0]
            cY = cent_warped[1]
            cZ = cent_warped[2]
            time = times[i]            
            if i > 0 :
                cent_prev_warped = np.dot(centroids[i-1], M)
                dx = (cX - cent_prev_warped[0])
                dy = (cY - cent_prev_warped[1])
                dz = (cZ - cent_prev_warped[2])
                dist = np.sqrt(dx**2 + dy**2) * scale
            else: # We dont have information on previous location
                dist = 0;   
                
            f.write("{:d},{:0.2f},{:0.2f},{:0.2f},{:0.2f}\n".format(i, cX, cY,
                                                                    time, dist))
            total_dist += dist
            
        f.write("Total dist in mm: {:0.4f}\n".format(total_dist))
        f.write("Total time in sec: {:0.4f}\n".format(times[-1]))
    # f.close() is implicit

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
    pts = getcoords.order_points(pts)
    return pts

def tracker(prob_mask, bbox, type = "meanShift", dist_lim = p.dist_lim, **kwargs):
    """Tracks an object in image
    
    The function is to be extended at will to achieve reliable tracking of given
    object.
    
    Parameters
    ---------------
    type : str
        Defines the tracking method to use
    prob_mask : ndarray
        2D probability mask
    bbox : tuple
        Bounding box (minY, minX, deltaY, deltaX)
    
    Returns
    -----------
    bbox : ndarray
    pts : ndarray
    """
    if prob_mask is None:
        return None
       
    bbox_old = bbox
    pts_old = bbox_to_pts(bbox_old)
    
    if type == "meanShift":
        ret, bbox = cv2.meanShift(prob_mask, bbox, kwargs["term_crit"] )
        pts = bbox_to_pts(bbox)
        
    elif type == "CamShift":    
        ret, bbox = cv2.CamShift(prob_mask, bbox, kwargs["term_crit"])
        pts = cv2.boxPoints(ret)
            
    c_old = rectangle_to_centroid(pts_old)
    c = rectangle_to_centroid(pts)
    dist = np.sqrt((c[0] - c_old[0])**2 + (c[1] - c_old[1])**2)
    
    if dist < dist_lim:
        bbox = bbox_old
        pts = pts_old
    
    return (bbox, pts)

def check_min_area( prob_mask, frame, min_area = p.min_area, type = 2, 
                    annotate = p.annotate_mask, **kwargs):
    (frame_binary, cnts) = label_contours(  frame, type, **kwargs)    
    #prob_mask = cv2.bitwise_and(prob_mask, prob_mask, mask= frame_binary)

    num_nonzero = np.count_nonzero(frame_binary)
    if num_nonzero < min_area: 
        prob_mask = None
    
    if annotate:
        w = (255, 255, 255)
        cnt_metric = cv2.arcLength(cnts[0], True)
        ann_str = "Max perim: {:.2f}, #Contours: {}, #Nonzero: {}"\
                    .format(cnt_metric, len(cnts), num_nonzero)
        ann_loc = (50, 50)
        cv2.putText(frame_binary, ann_str, ann_loc, cv2.FONT_HERSHEY_PLAIN, 3, w)
    
    return prob_mask, frame_binary
                                              
def label_contours(frame, type = 2 , **kwargs):
    """Find binary image of object of interest
    
    Facilitate object tracking by first thresholding and creating
    a binary mask
    
    Parameters
    ----------
    frame : ndarray
        average 3 channel frame
    type : int
        flag indicating which approach to use
        
    Returns
    ----------
    vis_image : 
        binary mask of found object
    cnts :
        found contours
    """
    kSize_gauss = kwargs["kSize_gauss"]
    sigmaX = kwargs["sigmaX"]
    kSize_canny = kwargs["kSize_canny"]
    
    # frame = cv2.GaussianBlur(frame, kSize_gauss, sigmaX)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kSize_canny)
    vis_img = np.zeros(frame.shape[:2], dtype = np.uint8)

    if type == 1:
        min_area = p.min_area
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.uint8)
        edged = cv2.Canny(frame, threshold1 = 120, threshold2 = 255)
        edged = cv2.dilate(edged, kernel = kernel, iterations = 2)
        
        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)[1]
        cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]
        cnts = [cv2.convexHull(cnt) for cnt in cnts if cv2.contourArea(cnt) > min_area]
        
        cv2.drawContours(vis_img, cnts, contourIdx = -1, 
                        color = (255, 255, 255), thickness =  -1)
        vis_img = cv2.morphologyEx(vis_img, cv2.MORPH_CLOSE, kernel, iterations =1)
        
    elif type == 2:
        fraction = p.fraction
        connectivity = p.connectivity
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.uint8)
        binary = np.where(frame > np.max(frame) * fraction, 255, 0).astype(np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations =3)
        
        ccs = cv2.connectedComponentsWithStats(binary, connectivity, cv2.CV_32S)
        labels = ccs[1]
        # ccs = (num_labels, labels, stats, centroids)
        unique, counts = np.unique(labels, return_counts = True)
        pair_list = [(lab, num) for (lab, num) in zip(unique, counts)]
        ranking = sorted(pair_list, key = lambda x: x[1], reverse = True)
        label = ranking[1][0]
        
        vis_img[labels == label] = 255
        # vis_image = cv2.bitwise_and(binary, binary, mask = mask)
        cnts = cv2.findContours(vis_img.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)[1]                               
                        
    return (vis_img, cnts)

def segment_background(frame, pts, y_pad = 0):
    """Substract stationary background
    
    """
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(frame_gray)
    bbox = cv2.boundingRect(pts)
    pts = swap_coords_2d(pts)
    mask[pts[0][0]:pts[1][0], pts[0][1]:pts[2][1] + y_pad] = 1
    
    frame_corrupt = frame.copy()
    frame_corrupt[mask == True] = 0
    frame_inpaint = cv2.inpaint(frame_corrupt, mask, inpaintRadius = 5,
                                flags = cv2.INPAINT_NS)
    
    bgd_model = np.zeros((1,65), dtype = np.float64)
    fgd_model = np.zeros((1,65), dtype = np.float64)
    frame_segment = frame.copy()
    mask_segment = np.zeros(frame.shape[:2],np.uint8)
    cv2.grabCut(frame_segment, mask_segment, bbox, bgd_model, fgd_model,
                iterCount = 3, mode = cv2.GC_INIT_WITH_RECT)
    
    # 1 = obvious FGD, 3 = Possible FGD
    mask_fgd = np.where((mask_segment==1) | (mask_segment==3),1,0).astype('uint8')
    #img = cv2.bitwise_and(frame, frame, mask= mask_fgd)
    #frame_segment[mask_fgd == 0, :] = 0

    return frame_inpaint, mask_fgd

def subtract_background(frame, background, mask_fgd = np.empty(0)):
    """Subtract background level from ROI image
    """
    if frame.dtype.type is not np.uint8:
        frame = frame.astype(np.uint8)
    if mask_fgd.size: #if exists
        mask = np.invert(mask_fgd, dtype = np.uint8)
        frame_bg_removed = cv2.subtract(background, frame, mask = mask)
    else:
        frame_bg_removed = cv2.subtract(background, frame)
   
    cv2.normalize(  src = frame_bg_removed, dst = frame_bg_removed,
                alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX)
    
    return frame_bg_removed

def average_frames(vid, frame_avg, step = 5, alpha = 0.5, frame_count = 0):
    """Running average of frames from video stream
    """
    frame_count += 1 * step
    frame_avg = frame_avg.astype(np.float32)
    for i in range(step):
        # Running average
        ret, frame = vid.read()
        if frame is None: break; # if stack empty, break
        frame = frame.astype(np.uint8)
        # Accumulate running average with alpha weighting
        cv2.accumulateWeighted(src = frame, dst = frame_avg, alpha = alpha)
    return frame_avg, frame_count, frame

def initialize_tracking(video_src, remove_bg, skip_init = False, save_vars = True):
    vid = cv2.VideoCapture(video_src)
    fname = "init.npz"
    base_skip = 400
    roi_pad = 250
    remove_bg = True
    
    if not skip_init:
        (vid, _) = getcoords.go_to_frame(vid, base_skip, video_src,
                            return_frame = False)
        pts, roi, vid, frame_pos, frame = \
                    getcoords.select_roi_video(vid, release = False, show_frames = True)
        # Convert between different rectangle representations
        pts = swap_coords_2d(pts)
        bbox = cv2.boundingRect(pts)

        if remove_bg : 
            background, mask_fgd = segment_background(frame, pts, p.y_pad)
            frame_bg_removed = subtract_background(frame, background)
            roi = frame_bg_removed[pts[0][0]:pts[1][0], pts[0][1]:pts[2][1],:]
        else :
            background = np.empty(0);

        roi_hist, sizes, ranges, chs = get_roi_hist(roi, vid, background, frame_pos,
                                                        reinitialize = True)
        if save_vars:
            
            np.savez_compressed(fname, 
                                roi_hist = roi_hist,
                                pts = pts,
                                background = background,
                                sizes = sizes,
                                ranges = ranges,
                                chs = chs,
                                frame_pos = frame_pos)
    
    if skip_init:
        (vid, frame) = getcoords.go_to_frame(vid, base_skip, video_src,
                            return_frame = True)
        vid, frame_pos, frame = preprocess.find_frame_video(video_src)
        
        try:
            loaded_vars = np.load(fname)
        except:
            print("no preset available, exiting"); sys.exit()
            
        roi_hist = loaded_vars["roi_hist"]
        pts = loaded_vars["pts"]
        background = loaded_vars["background"]
        sizes = loaded_vars["sizes"]
        ranges = loaded_vars["ranges"]
        chs = loaded_vars["chs"]
        frame_pos = loaded_vars["frame_pos"]
    
    pts[:, 0] = pts[:, 0] - roi_pad
    pts[:, 1] = pts[:, 1] + roi_pad
    bbox = cv2.boundingRect(pts)    

    return vid, roi_hist, pts, bbox,  background, frame, sizes, ranges, chs, frame_pos

def make_filename(full_path, ext_new = "", init_fname = None):
    """"Constructs filename from given path and extension
    """
    (path, _) = os.path.splitext(full_path)
    (head, tail) = os.path.split(path)
    if ext_new: ext = ext_new;
    if init_fname:
        fname = os.path.normpath(head) + "\\" + init_fname + ext
    else:
        fname = os.path.normpath(path) + ext
    return fname
    
def save_tracking_params(video_src, save_dict, ext, init_fname = None):
    save_name = make_filename(video_src, ext, init_fname)
    if ext == ".dat":
        with open (save_name, 'wb') as outfile:
            pickle.dump(save_dict, outfile, protocol = pickle.HIGHEST_PROTOCOL)
    elif ext == ".npz":
        np.savez_compressed(fname, save_dict)

def load_tracking_params(base, ext, names, init_fname = None):
    load_name = make_filename(base, ext, init_fname)
    if ext == ".dat":
        with open(load_name, "rb") as infile:
            loaded = pickle.load(infile)
    elif ext == ".npz":
        loaded = np.load(fname)
    
    dict_out = dict((k , loaded[k]) for k in names)
    return dict_out
    
def get_parameter_names(remove_bg, reinitialize_hsv, reinitialize_roi,
                        reinitialize_bg):
    load_bg = not reinitialize_bg
    load_hsv = not reinitialize_hsv
    load_roi = not reinitialize_roi
    
    names_init = []
    names = []
    
    if remove_bg and load_bg:
        names.append("background")
        names_init.append("background")
        
    if load_hsv:
        names.append("roi_hist")
        names_init.append("roi_hist")
        
    if load_roi:
        names.append("pts")
        
        
    return names, names_init
    
def adjust_flags(init_flag, remove_bg, reinitialize_roi, reinitialize_hsv, save_init):
    if init_flag:
        save_init = True
        init_fname = "init"
    else:
        save_init = False
        reinitialize_bg = False
        reinitialize_hsv = False
        init_fname = None
        
def get_in_out_names(video_src, init_flag, save_init):
    data_out = video_src
    param_in = video_src
    
    if init_flag and save_init:
        init_out = make_filename(video_src, "", "init")
    elif init_flag:
        init_in = None
    else:
        init_out = None
        init_in = make_filename(video_src, "", "init")
        
    fnames = [data_out, param_in, init_out, init_in]
    return fnames
            
    
def track_motion(   video_src, init_flag = False,
                    remove_bg = p.remove_bg, 
                    reinitialize_roi = p.reinitialize_roi,
                    reinitialize_hsv = p.reinitialize_hsv,
                    reinitialize_bg = p.reinitialize_bg,
                    save_init = p.save_init):
    
    startdebug()
    (pnames, pnames_init) = get_parameter_names(remove_bg, reinitialize_hsv,                                            reinitialize_roi, reinitialize_bg)
    fnames = get_in_out_names(video_src, init_flag, save_init)
    
    try:
        if pnames:
            p_vars_curr = load_tracking_params(fnames[1], p.ext, pnames)
        if pnames_init and init_flag:
            p_vars_init = load_tracking_params(fnames[3], p.ext, pnames_init)
        
        p_vars = {**p_vars_curr, **p_vars_init}
        
    except:
        print("Some parameters couldn't be loaded")
    
    if reinitialize_roi:
        pts, _, vid, frame_pos, frame = getcoords.select_roi_video(video_src)
        # Convert between different rectangle representations
        pts = swap_coords_2d(pts)
    else:
        # pts, roi, vid, frame_pos, frame = from_preset(video_src)
        pts = p_vars["pts"]
        frame_pos = p_vars["frame_pos"]
        (vid, frame) = getcoords.go_to_frame([], frame_pos, video_src, return_frame = True)
    
    
    if remove_bg and reinitialize_bg:
        background, _ = segment_background(frame, pts, p.y_pad)
    elif remove_bg and not reinitialize_bg:
        background = p_vars["background"]
        
    if remove_bg:
        frame_bg_removed = subtract_background(frame, background)
        roi = frame_bg_removed[pts[0][0]:pts[1][0], pts[0][1]:pts[2][1],:]
    else:
        background = np.empty(0);
        roi = frame[pts[0][0]:pts[1][0], pts[0][1]:pts[2][1],:]
    
    if reinitialize_hsv:
        roi_hist = get_roi_hist(roi, vid, background, frame_pos,
                                reinitialize_hsv)
    else:
        roi_hist = p_vars["roi_hist"]
    
    bbox = cv2.boundingRect(swap_coords_2d(pts))                                            
    # vid, roi_hist, pts, bbox, background, frame, sizes, ranges, chs, frame_pos = \
                # initialize_tracking(video_src, remove_bg, skip_init = True, save_vars = True)
    # debug_plot2(frame, pts, [])
    # startdebug()
    
    save_flags = [reinitialize_roi, reinitialize_hsv, reinitialize_bg]
    if save_init and any(save_flags):
        names, names_init = get_parameter_names(remove_bg, not reinitialize_hsv,
                                    not reinitialize_roi, not reinitialize_bg)
        local_variables = locals()
        save_dict = dict((n, local_variables[n]) for n in names)
        save_tracking_params(fnames[0], save_dict, p.ext)
        
        save_dict_init = dict((n, local_variables[n]) for n in names_init)
        save_tracking_params(fnames[2], save_dict_init, p.ext)
    
    # Setup the termination criteria, either 10 iteration or move by atleast 1 pt
    term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, p.its, p.eps )
    
    fps = vid.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    # Lists to temporalily store output data
    times = []
    centroids = []
    
    # debug_plot2(frame, pts)
    # startdebug()
    while vid.isOpened():
                   
        frame_avg = np.zeros_like(frame, dtype = np.float32)
        frame_avg, frame_count, frame = average_frames( vid, frame_avg, p.step,
                                                        p.alpha, frame_count)    
        # Break out of this loop if emptied stack      
        if frame is None: print ("End of stream"); break;

        if remove_bg:
            frame_avg = subtract_background(frame_avg, background)
        
        params = dict(  kSize_gauss = p.kSize_gauss, sigmaX = p.sigmaX,
                        kSize_canny = p.kSize_canny, term_crit = term_crit)
                        
        prob_mask, frame_binary = prob_mask_hsv(frame_avg, roi_hist, p.h_ranges,
                                                p.chs, **params)
        
        res = tracker(  prob_mask, bbox, "meanShift", **params)
        
        if (res is None) or prob_mask is None:
            print("No components detected, end of tracking")
            vid.release()
            break;
            
        bbox = res[0]
        pts = res[1]
        if len(res) > 2:
            cent = res[2]
        else:
            cent = rectangle_to_centroid(pts)
            
        centroids.append(cent)
        time = float(frame_count)/fps
        times.append(time) #Save time 
        
        # Visualize
        if p.show_frames:
            # frame_vis = prob_mask.copy() # frame_avg.copy()
            frame_vis = cv2.cvtColor(frame_avg, cv2.COLOR_BGR2GRAY).astype(np.uint8)
            
            (r, b, w) = ((0, 0, 255), (0, 0, 0), (255, 255, 255))
            # Put timestamp on the average image
            time_str = "{:.2f}".format(time)
            dimy, dimx = frame_vis.shape[:2]
            time_loc = (int(dimx-250), dimy-150)
            cv2.putText(frame_vis, time_str, time_loc, cv2.FONT_HERSHEY_PLAIN, 5, w)
            prob_str = "Max prob: {:.2f}%".format(np.max(prob_mask) / 2.55)
            prob_loc = (50, 50)
            cv2.putText(frame_vis, prob_str, prob_loc, cv2.FONT_HERSHEY_PLAIN, 3, w)
            cv2.polylines(frame_vis, pts =[pts], isClosed= True, color= w, thickness= 2)
            # Draw location of center of mass on the average image
            cv2.circle(frame_vis, tuple(cent)[:2], radius = 4, color = b, thickness = 4)
            
            # Image should be uint8 to be drawable in 0, 255 scale
            # https://stackoverflow.com/questions/9588719/opencv-double-mat-shows-up-as-all-white
            (frame_vis, _) = getcoords.resize_frame(frame_vis, height = p.height_resize)
            cv2.imshow("Tracking", frame_vis)
            if p.plot_mask: 
                (frame_binary, _) = getcoords.resize_frame(frame_binary, height = p.height_resize)
                cv2.imshow("Mask", frame_binary)
                    
            # Interrup on ESC
            ch = 0xFF & cv2.waitKey(1)
            if ch == 27: break;
            elif ch == ord('d'): startdebug()
            
    # end while True:
    output_data(centroids, times, video_src)
    cv2.destroyAllWindows()

def main():
    # Read video file path from user input
    try: 
        video_src = ap.parse_args().video
        init_fname = ap.parse_args().init_fname
    except: 
        video_src = 0
        init_flag = None
    # Show documentation
    # print(__doc__)
    # Assure that all windows get destroyed at the end of the run
    try:
        track_motion(video_src, init_fname)
    finally:
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
