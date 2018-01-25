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
import getcoords
import sys
import utils
import os

# Parse arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type = str, help= "path to the video file")
ap.add_argument("-fr", "--frame_range", type = str, help ="[sart_frame, stop_frame]",
                default = [])
ap.add_argument("-p", "--params", type = str, help= "parameter set", 
                default = "params")
try:
    args = ap.parse_args()
    video_src = args.video
    parameter_set = args.params
    frame_range = utils.convert_list_str(args.frame_range) 
except Exception as ex: 
    template = "An exception of type {0} occurred. Arguments:\n{1!r}"
    message = template.format(type(ex).__name__, ex.args)
    print(message)
    
    parameter_set = "params"
    frame_range = []
    video_src = []

init_flag = True if parameter_set == "params_init" else False

if init_flag:
    import params_init as p
else: 
    import params as p
    

def set_review_flags(dists, num_nonzeros, times):
    """Sets flags for supervised control
    
    Tracking may fail due to multiple reasons and often manifests itself by any of: 
    1) detected object being stationary for too long period of time, 2) travelled 
    distance between consecutive frames unepxectedly large or 3) changes in illumination
    that persist over multiple consecutive frames.

    Parameters
    ----------
    dists : list
        traveled distances corresponding to all consecutive (constant) time deltas
    num_nonzeros : int
        number of oonzero pixels in binary representation of foreground and background
        (fg = 1, bg = 0)
    times : 
        list of all times at which frames are collected

    Returns
    ----------
    max_num_static : int
        maximum number of frames over which dtetected object didn't move
    max_num_large : int
        maximum number of frames over which object area was unexpectedly large
        (possibly due to persistent change of illumination)
    max_dist : int
        maximal distance travelled by object over all time-deltas

    References
    ------------
    segment_stationary_background, label_contours
    """
    from itertools import groupby
    
    #1
    gradients = np.around(np.gradient(dists)) # alternative: np.diff(dists)
    group_reps = [[len(list(vals)), key] for (key, vals) in groupby(gradients)]
    group_reps = np.asarray(group_reps)
    
    try:
        max_num_static = np.max(group_reps[group_reps[:,1] == 0,0])
    except ValueError: # Object never static
        max_num_static = 0
    
    #2        
    thresh = 40000
    num_nonzeros_thresh = [True if x > thresh else False for x in num_nonzeros]
    group_reps = [[ len(list(vals)), key] for (key, vals) in 
                    groupby(num_nonzeros_thresh)]
    group_reps = np.asarray(group_reps)
    if group_reps.size & any(group_reps[:, 1] == True):
        max_num_large = np.max(group_reps[group_reps[:, 1] == True,0])
    else: # Never above threshold
        max_num_large = 0;
    
    #3
    diffs = np.diff(dists)
    diffs  = np.sort(diffs)
    max_dist = np.abs(diffs[0])
    
    return max_num_static, max_num_large, max_dist
    
def flag_file_review(f, dists, num_nonzeros, times):
    """Marks file for manual control by appending warnings to output
    
    Parameters
    ----------
    f : file handle
        handle to file to write warnings into
    dists : list
        traveled distances corresponding to all consecutive (constant) time deltas
    num_nonzeros : int
        number of oonzero pixels in binary representation of foreground and background
        (fg = 1, bg = 0)
    times : 
        list of all times at which frames are collected
    
    Returns
    ------------
    f : file handle
        handle to file with flags appended
        
    References
    ---------------
    set_review_flags
    """

    max_rep_count, max_large_count, max_dist = \
                            set_review_flags(dists, num_nonzeros, times)

    if max_rep_count > 30:
        f.write("WARNING:NoMotion: Object static for {:d}*`step` frames\n". \
        format(np.int(max_rep_count)))
    if max_large_count > 30:
        f.write("WARNING:Ilumination: Large object area for {:d}*`step` frames\n". \
        format(np.int(max_large_count)))
    if max_dist > 150:
        f.write("WARNING:Distance: Too large movement of {:d} pixels\n". \
        format(np.int(max_dist)))
        
    return f
    
def compute_distance(centroids, cent, i, M, scale):
    """Compute distance between two centroids
    
    Uses precomputed transformations parameters.
    
    Parameters
    -----------
    i : int
        iterator, function is called in loop
    M : ndarray
        3x3 projective transformation matrix
    scale : float
        scaling factor in units mm / pixel
        
    References
    -------
    getcoords.projective_transform, getcoords.find_scale
    """
    cent_warped = np.dot(cent, M)
    cX = cent_warped[0]
    cY = cent_warped[1]
    cZ = cent_warped[2]          
    if i > 0 :
        cent_prev_warped = np.dot(centroids[i-1], M)
        dx = (cX - cent_prev_warped[0])
        dy = (cY - cent_prev_warped[1])
        dz = (cZ - cent_prev_warped[2])
        dist = np.sqrt(dx**2 + dy**2) * scale
    else: # We dont have information on previous location
        dist = 0;
    
    return dist, cX, cY, cZ
  
def output_data(centroids, times, num_nonzeros, video_src):
    """Writes select data into text file
    
    Saves centroid coordinates, traveled distance and elapsed time to file. 
    output_data is called in postprocessing. 
    
    Parameters
    -----------
    centroids : list
        list of centroids of tracked object sampled at times
    times : list
        List of sampling times. Sampling every step-th frame.
    num_nonzeros : list
        List of total number of nonzero pixel values in binary foreground mask 
        for each time step
        
    References
    -----------
    getcoords.find_scale, getcoords.projective_transform, set_review_flags
    """
    # Initalize variables that hold sums
    total_dist = 0
    dists = np.empty(len(centroids))
    rep_counter = 0
    max_rep_count = 0
    
    scale = np.load("src\\scaling_factor.npy")
    M = np.load("src\\projection_matrix.npy")
    
    fname_out = utils.make_filename(video_src, ".txt", parent = "res")
    
    try:
        os.mkdir("res")
    except FileExistsError:
        pass
    
    with open(fname_out, "w") as f:
        f.write("No.,cX,cY,time,dist\n")
        for i, cent in enumerate(centroids):
            #Skip point if we have lost track
            if  np.any(cent[:2] == 0) or (i > 0 and np.any(centroids[i-1][:2] == 0)):
                #f.write("#Discarding line! {:0.2f},{:0.2f} -> {:0.2f},{:0.2f}\n".format(centroids[i-1][0], centroids[i-1][1], cent[0], cent[1]))
                dist = np.nan
                continue;
            
            time = times[i]                       
            dist, cX, cY, cZ = compute_distance(centroids, cent, i, M, scale)
                
            f.write("{:d},{:0.2f},{:0.2f},{:0.2f},{:0.2f}\n".format(i, cX, cY,
                                                                    time, dist))
            total_dist += dist if dist is not np.nan else 0
            dists[i] = dist
        
        f.write("Total dist in mm: {:0.4f}\n".format(total_dist))
        f.write("Total time in sec: {:0.4f}\n".format(times[-1]))
        
        f = flag_file_review(f, dists, num_nonzeros, times)     
    # f.close() is implicit

def average_frames(vid, frame_avg, step = 5, alpha = 0.5, frame_count = 0):
    """Running average of frames from video stream
    
    Parameters:
    ---------------
    step : int
        number of frames to average
    alpha : int 
        relative weight of previous frames
    frame_count : int
        current frame number, relative to start of tracking
    
    Returns
    ------------------
    frame_avg : np.ndarray
        accumulated weighted averge of frames
    frame_count : int
    frame : np.ndarray
        current frame
    """
    frame_count += 1 * step
    frame_avg = frame_avg.astype(np.float32)
    
    for i in range(step):
        ret, frame = vid.read()
        if frame is None: break;
        frame = frame.astype(np.uint8)
        # Accumulate running average with alpha weighting
        cv2.accumulateWeighted(src = frame, dst = frame_avg, alpha = alpha)
        
    return frame_avg, frame_count, frame
    
def find_frame_video(   video_src, show_frames = False, release = False):
    """Select a frame from arbitrary video.
    
    Prompts the user to select a frame in video.
    
    Parameters
    ----------
    video_src : str
        path to video, must be in compatible format (avi is OK)
    release : bool
        a flag indicating whether to release video object
    
    Returns
    ---------
    pts : ndarray
        4x2 array of box points
    roi : ndarray
        2D image cropped to pts
    vid : opencv video object
        video captured from video_src
    frame_pos : float
        number of frame corresponding to current position in video
        
    References
    ----------------
    getcoords.select_roi
    """
    # Prompt user to select a good frame we will work on later
    print ("Press key `p` to select the ROI")
    # Capture video
    try:
        while vid.isOpened():
            # Read next frame
            retval, frame = vid.read()
            # Exit when video can't be read
            if not retval: print ('Cannot read video file'); sys.exit();
            
            # Pause on 'p'
            if(cv2.waitKey(1) == ord('p')):
                break # Break from while loop
            if show_frames:
                # Show the current frame
                cv2.namedWindow("SelectROI", cv2.WINDOW_NORMAL)
                cv2.imshow("SelectROI", frame)
    finally:
        # Assure all windows closed
        cv2.destroyAllWindows()
        
    # Get posiiton in videos in frames
    frame_pos = vid.get(cv2.CAP_PROP_POS_FRAMES)
    # Optionally release video object
    if release: vid.release();
        
    return vid, frame_pos, frame
     
def get_roi_hist(   roi_rgb, vid, background = np.empty(0), frame_pos = [], 
                    reinitialize = False, normalize = True):
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
    
    References
    -----------------
    ..[1] http://docs.opencv.org/3.3.0/dd/d0d/tutorial_py_2d_histogram.html
    """
    video_src = []
    if type(vid) is str:
        video_source = vid
        vid = []
    
    (hsv_lowerb, hsv_upperb) = select_hsv_range(vid, video_src, background,
                                                frame_pos, reinitialize)
    
    chs = [0, 1, 2]
    h_sizes = [180, 256, 256]
    h_ranges = [(0, 179),  (0, 255), (0, 255)]
    
    diffs = hsv_upperb - hsv_lowerb + 1
    update_tf = [x != y for x,y in zip(diffs, h_sizes )]
    chs = [ch for tf,ch in zip(update_tf, chs) if tf == True]
    h_sizes = [hs for tf,hs in zip(update_tf, h_sizes) if tf == True]
    h_ranges = [x for tf,hr in zip(update_tf, h_ranges) if tf == True for x in hr]
    
    roi_hsv = cv2.cvtColor(roi_rgb, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(roi_hsv, hsv_lowerb , hsv_upperb)
    roi_hist = cv2.calcHist(images = [roi_hsv], channels = chs, mask = mask,
                            histSize = h_sizes, ranges = h_ranges)                       

    if normalize:
        cv2.normalize(  src = roi_hist, dst = roi_hist,  alpha = 0, beta = 255,
                        norm_type = cv2.NORM_MINMAX)
    return roi_hist, h_ranges, chs

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
                    
        frame, _ = utils.resize_frame(frame, height = p.height_resize)
                
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
            hsv_lowerb = np.array([0, 0, 150]) # with bg substraction
            hsv_upperb = np.array([179, 50, 255]) # with bg substraction
        
    return (hsv_lowerb, hsv_upperb)

def segment_background(frame, bbox, **kwargs):
    """Supervised background subtraction
    
    Uses user-defined bounding box to inpaint frame and create estimate of
    background.
    
    Parameters
    -------------
    frame : np.ndarray
    bbox : tuple
    
    Returns
    --------------
    frame_inpaint: np.ndarray
        background estimate
    mask_fgd: np.ndarray
        binary mask of foreground pixels
    """
    y_pad = 0 # optional padding of a roi
    kSize_gauss =  (21, 21) #kwargs["kSize_gauss"]
    sigmaX = 12 #kwargs["sigmaX"]
    
    mask = np.zeros(frame.shape[:2], np.uint8)
    (c, r ,w, h) = bbox
    mask[r:r+h, c: c+w] = 1
      
    frame_corrupt = frame.copy()
    frame_corrupt[mask == True] = 0
    frame_inpaint = cv2.inpaint(frame_corrupt, mask, inpaintRadius = 5,
                                flags = cv2.INPAINT_NS)
    
    frame_inpaint = cv2.GaussianBlur(frame_inpaint, kSize_gauss, sigmaX)
    
    bgd_model = np.zeros((1,65), dtype = np.float64)
    fgd_model = np.zeros((1,65), dtype = np.float64)
    frame_segment = frame.copy()
    mask_segment = np.zeros(frame.shape[:2],np.uint8)
    cv2.grabCut(frame_segment, mask_segment, bbox, bgd_model, fgd_model,
                iterCount = 3, mode = cv2.GC_INIT_WITH_RECT)
    
    # 1 = obvious FGD, 3 = Possible FGD
    mask_fgd = np.where((mask_segment==1) | (mask_segment==3),1,0).astype('uint8')
        
    return frame_inpaint, mask_fgd

def subtract_background(frame, background, mask_fgd = np.empty(0), normalize = True):
    """Subtract background level from ROI image
    
    Note: For this particular application, we substract frame from background
    
    Parameters
    --------------
    frame : np.ndarray
    background : np.ndarray
    mask_fgd : np.ndarray
        2D, binary. [default = np.empty(0)]
    normalize : bool
        flag indicating whether to normalize pixel values between 0 and 255 
        [default = True]
    
    Returns
    -------------
    frame_bg_removed : np.ndarray
        frame with background subtracted
        
    References
    -----------
    ..[1] https://pythonprogramming.net/grabcut-foreground-extraction-python-opencv-tutorial/
    """
    if frame.dtype.type is not np.uint8:
        frame = frame.astype(np.uint8)
    if mask_fgd.size: #if exists
        mask = np.where(mask_fgd > 0, 0, 255).astype(np.uint8)
        frame_bg_removed = cv2.subtract(background, frame,  mask = mask)
    else:
        frame_bg_removed = cv2.subtract(background, frame)
        
    if normalize:
        cv2.normalize(  src = frame_bg_removed, dst = frame_bg_removed,
                        alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX)
    
    return frame_bg_removed
       
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
    
    return prob_mask

def check_min_area( prob_mask, frame, min_area = p.min_area, type = 2, 
                    check_area = p.check_area, **kwargs):
    """Finds contours in image and optinally checks if area bigger than `min_area`
    
    Parameters
    ------------
    prob_mask : ndarray
        2D probability map output of `prob_mask_hsv`
    min_area : int
        required minimal number of pixels in filled contours
    type : int
        type of binarization of input. 1 = Canny, 2 = Thresholding
    check_area : bool
        Flag indicating whether to check if area large enough
    **kwargs : dict
        parameters passed to `label_contours` function
    
    Returns
    -------------
    prob_mask : ndarray
    frame_binary : ndarray
        binary mask of found objects
    """
    (frame_binary, cnts) = label_contours(  frame, type, **kwargs)    
    #prob_mask = cv2.bitwise_and(prob_mask, prob_mask, mask= frame_binary)

    num_nonzero = np.count_nonzero(frame_binary)
    if (num_nonzero < min_area) and check_area: 
        prob_mask = None
    
    return prob_mask, frame_binary
                                              
def label_contours(frame, type = 2, dark = False, **kwargs):
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
    if frame.ndim > 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.uint8)

    if type == 1:
        min_area = p.min_area
        
        edged = cv2.Canny(frame, threshold1 = 20, threshold2 = 255)
        edged = cv2.dilate(edged, kernel = kernel, iterations = 2)
        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)[1]
        cnts = sorted(  cnts, key = lambda x: cv2.arcLength(x, closed = True), 
                        reverse = True)[:1]
        cnts = [cv2.convexHull(cnt) for cnt in cnts if cv2.contourArea(cnt) > min_area]
        
        cv2.drawContours(vis_img, cnts, contourIdx = -1, 
                        color = (255, 255, 255), thickness =  -1)
        vis_img = cv2.morphologyEx(vis_img, cv2.MORPH_CLOSE, kernel, iterations =1)
        #vis_img = cv2.dilate(vis_img, kernel = kernel, iterations = 2)
        
    elif type == 2:
        fraction = p.fraction
        connectivity = p.connectivity
        if dark:
            fraction = min(2*fraction, 1.0)
            binary = np.where(frame < np.max(frame) * fraction, 255, 0).astype(np.uint8)
        else:
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
        if frame.ndim > 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.uint8)
            
    elif type == 3:
        pass
    return (vis_img, cnts)
    
def subtract_stationary_background(fgbg, frame_avg, frame_count, **kwargs):
    """Fit mixture of gaussians as background model
    
    Parameters
    ------------
    fgbg : cv2 MOG2 background subtractor object
        used settings are {history=200, varThreshold=12, detectShadows=False,
        complexityReductionThreshold = 0.2, backgroundRation = 1}
    frame_avg : ndarray
        3channel RGB image of the whole frame, optinoally averaged over several
        frames
    frame_count : int
        Number of current frame, multiple of `step`
    **kwargs : dict
        other arguments passed as key,value pairs
        
    Returns
    -----------
    frame_avg : ndarray
        average frame with background subtracted
    frame_fgd : ndarray
        binary mask indicating foreground pixels
    cnts : list
        List of points describing contours found in the image
    
    References
    -----------
    ..[1] https://docs.opencv.org/3.3.0/db/d5c/tutorial_py_bg_subtraction.html
    """
    kSize_canny = kwargs["kSize_canny"]
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, kSize_canny)
    learn_rate = max(0.04, 1/fgbg.getHistory())
    frame_fgd = fgbg.apply(frame_avg, learningRate = learn_rate)
    #bg_ratio  = fgbg.getBackgroundRatio()
    #crt = fgbg.getComplexityReductionThreshold()
    cnts = []
    
    if fgbg.getDetectShadows(): #ignore shadows detection
        frame_fgd = np.where(frame_fgd > 0, 255, 0).astype(np.uint8)
    
    frame_fgd = cv2.dilate(frame_fgd, kernel = kernel, iterations = 1)
    frame_fgd = cv2.erode(frame_fgd, kernel = kernel, iterations = 2)
    frame_fgd = cv2.dilate(frame_fgd, kernel = kernel, iterations = 5)
    (frame_fgd, cnts) = label_contours(frame_fgd, type = 1, **kwargs)
    
    if frame_count > 0:
        new_bg = fgbg.getBackgroundImage()
        frame_avg = subtract_background(new_bg, frame_avg, normalize = True)
    else:
        cv2.normalize(  src = frame_avg, dst = frame_avg,
                alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX)
                
    return frame_avg , frame_fgd, cnts
    
def tracker(prob_mask, bbox, type = "meanShift", dist_lim = p.dist_lim, **kwargs):
    """Tracks an object in image
    
    `meanShift` type keeps constant size of bbox, whereas `camShift` adjusts it.
    The former is more reliable.
    
    Usage notes:
        - Make bbox significantly bigger than your object (~5x)
        - The bbox should be larger in direction of anticipated movement
        - Reliability may be increased purely by adding cv2.waitKey(n) for couple
          of initial frames processed. This may have something to do with speed of
          PC the algorith runs on.
    
    Parameters
    ---------------
    type : str
        Defines the tracking method to use
    prob_mask : ndarray
        2D probability mask
    bbox : tuple
        Bounding box (minY, minX, deltaY, deltaX)
    dist_lim : int
        optional minimal distance that the bbox is required to move between 
        time-steps in order for its position to be updated [default = 0]
    
    Returns
    -----------
    bbox : tuple
        updated bbox as tuple of 4 -> (x, y, dx, dy), where x horizontal,
        y vertical with origin at top-left
    pts : ndarray
        equivalent bbox but described by 4 vertices 
    
    References
    -----------
    .. [1] http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_video/py_meanshift/py_meanshift.html
    """
    if prob_mask is None:
        return None
       
    bbox_old = bbox
    pts_old = utils.bbox_to_pts(bbox_old)
    
    if type == "meanShift":
        ret, bbox = cv2.meanShift(prob_mask, bbox, kwargs["term_crit"] )
        pts = utils.bbox_to_pts(bbox)
        
    elif type == "CamShift":    
        ret, bbox = cv2.CamShift(prob_mask, bbox, kwargs["term_crit"])
        pts = cv2.boxPoints(ret)
            
    c_old = utils.rectangle_to_centroid(pts_old)
    c = utils.rectangle_to_centroid(pts)
    dist = np.sqrt((c[0] - c_old[0])**2 + (c[1] - c_old[1])**2)
    
    if dist < dist_lim:
        bbox = bbox_old
        pts = pts_old
    
    return (bbox, pts)

def update_bbox_location(frame, bbox, **kwargs):
    """Update boundig box location
    
    Fits bounding box to found contours within previous bbox. This is done to enable
    general starting location while allowing for small deviations of initial 
    posiiton of the tracked object.
    
    Parameters
    -----------
    frame : ndarray
        current frame, can be 1 or 3 channels
    bbox : tuple
        current bounding box
    **kwargs : dict
        other arguments passed as key,value pairs
        
    Returns
    ------------
    bbox_new : tuple
        adjusted bbox (used for BG subtraction)
    bbox_new_pad : tuple
        adjusted bbox additionally paded to yet increase robustness (used for tracking)
    
    References
    -----------
    segment_background, tracker
    """
    (c, r ,w, h) = bbox     
    pad = kwargs["padding"]

    c_p = c - pad; r_p = r - pad
    w_p = w + 2*pad; h_p = h + 2*pad
    roi = frame[r_p : r_p + h_p, c_p: c_p + w_p, :]
    (vis_img, cnts) = label_contours(roi, type = 2, dark = True, **kwargs)
        
    bbox_new = cv2.boundingRect(cnts[0])
    (c_new, r_new, w_new, h_new) = bbox_new
    bbox_new = (c_new + c_p, r_new + r_p, w_new, h_new) # newly added
       
    c_new = c_new - pad + c_p; r_new = r_new - pad + r_p
    w_new = w_new + 2*pad; h_new = h_new + 2*pad
    bbox_new_pad = (c_new, r_new, w_new, h_new)

    return bbox_new, bbox_new_pad
   
def track_motion(   video_src, init_flag = False,
                    remove_bg = p.remove_bg, 
                    reinitialize_roi = p.reinitialize_roi,
                    reinitialize_hsv = p.reinitialize_hsv,
                    reinitialize_bg = p.reinitialize_bg):
                    
    """Track object in a video
    
    Implements background subtraction with GrabCut and MOG2 algorithms. Tracking is 
    based on either CamShift or meanShift algorithms.
    
    Parameters
    ---------------
    video_src : str
        relative path to video to be processed (in suitable format, e.g. .avi or .mkv,
        depends on PC and OS the algorith runs on)
    reinitialize_roi : bool
        [True for init, else False]
    reinitialize_hsv : bool
        [True for init, else False]
    reinitialize_bg : bool
        [True]
    init_flag : bool
        is this an initializing run? [True for init, else False]
    remove_bg : bool
        [default = True]
          
    Additional Parameters 
    (see available cmd line arguments and files params.py and params_init.py)
    -------------
    frame_range : list
        range of rames to process, read from csv file
    double_subtract_bg : bool
        use also MOG2? [default = True]
    params : str
        which parameter set to use
    height_resize : int
        height of image to plot, will be resized if different from current
    plot_mask : bool
        Flag indicating if binary fg mask should be plotted as well [default = True]
    annotate_mask : bool
        Add additional text information to plot_mask image? [default = True]
    show_frames : bool
        Plot frames from video and visualize tracking [default = True]
    save_frames : bool
        Flag indicating whether to save frames shown in `Tracking` window (cf. 
        show_frames), see utils.define_video_output
        
    Returns
    -----------
    None
    
    References
    ----------------
    ..[1] https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_video/py_bg_subtraction/py_bg_subtraction.html
    
    See also
    -----------
    process.py, params.py, params_init.py, utils.define_video_output
    """
    
    double_substract_bg = p.double_substract_bg
    save_init = any([reinitialize_roi, reinitialize_hsv, reinitialize_bg])
    params = dict(  kSize_gauss = p.kSize_gauss, sigmaX = p.sigmaX,
                    kSize_canny = p.kSize_canny, padding = p.padding)
    
    if double_substract_bg:
        fgbg = cv2.createBackgroundSubtractorMOG2(  history = 200, varThreshold = 12,
                                                    detectShadows = False)
        fgbg.setComplexityReductionThreshold(0.05*4)
        fgbg.setBackgroundRatio(1)
    (pnames, pnames_init) = utils.get_parameter_names(  remove_bg, reinitialize_hsv,                                                      reinitialize_roi, reinitialize_bg)
    fnames = utils.get_in_out_names(video_src, init_flag, save_init)

    try: 
        if pnames:
            p_vars_curr = utils.load_tracking_params(fnames[1], p.ext, pnames)
        else:
            p_vars_curr = {}
        if pnames_init and not init_flag:
            p_vars_init = utils.load_tracking_params(fnames[3], p.ext, pnames_init)
        else: p_vars_init = {}
           
        p_vars = {**p_vars_curr, **p_vars_init}
        
    except:
        print("Some parameters couldn't be loaded")
    
    if reinitialize_roi and not frame_range:
        pts, _, vid, frame_pos, frame = getcoords.select_roi_video(video_src)
        pts = utils.swap_coords_2d(pts)
    
    elif reinitialize_roi and frame_range:
        pts, _, vid, frame_pos, frame = getcoords.select_roi_video(video_src,
                                            frame_pos = frame_range[0])
        pts = utils.swap_coords_2d(pts)                                    
    elif not reinitialize_roi and frame_range:
        frame_pos = frame_range[0]
        (vid, frame) = getcoords.go_to_frame([], frame_pos, video_src, return_frame = True)
        pts = p_vars["pts"]      
    elif not frame_range:
        # pts, roi, vid, frame_pos, frame = from_preset(video_src)
        pts = p_vars["pts"]
        frame_pos = p_vars["frame_pos"]
        (vid, frame) = getcoords.go_to_frame([], frame_pos, video_src, return_frame = True)
    
    bbox = cv2.boundingRect(pts)
    (c, r ,w, h) = bbox
    bbox_min, bbox = update_bbox_location(frame, bbox, **params)
    (c, r ,w, h) = bbox

    if remove_bg and reinitialize_bg:
        background, mask_fgd = segment_background(frame, bbox_min, **params)
    elif remove_bg and not reinitialize_bg:
        background = p_vars["background"]
     
    if remove_bg:
        frame_bg_removed = subtract_background(frame, background)
        roi = frame_bg_removed[r:r+h, c: c+w, :]
    else:
        background = np.empty(0);
        roi = frame[r:r+h, c: c+w, :]
    
    if reinitialize_hsv:
        roi_hist, h_ranges, chs = get_roi_hist(roi, vid, background, frame_pos,
                                reinitialize_hsv)
    else:
        roi_hist = p_vars["roi_hist"]
        h_ranges = p_vars["h_ranges"]
        chs = p_vars["chs"]
                                                
    
    if save_init:
        names, names_init = utils.get_parameter_names(remove_bg, not reinitialize_hsv, not reinitialize_roi, not reinitialize_bg)
        local_variables = locals()
        if names:
            save_dict = dict((n, local_variables[n]) for n in names)
            _ = utils.save_tracking_params(fnames[0], save_dict, p.ext)
        if names_init and init_flag:
            save_dict_init = dict((n, local_variables[n]) for n in names_init)
            _ = utils.save_tracking_params(fnames[2], save_dict_init, p.ext)
    
    # Setup the termination criteria, either 10 iteration or move by atleast 1 pt
    term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, p.its, p.eps )
    params["term_crit"] = term_crit
    
    fps = vid.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    # Lists to temporalily store output data
    times = []
    centroids = []
    num_nonzeros = []
    stop_frame = frame_range[1] - frame_range[0]
    
    if p.save_frames:
        vid_out = utils.define_video_output(video_src, vid, fps, p.step, p.height_resize)
    
    while vid.isOpened() and frame_count <= stop_frame:
                   
        frame_avg = np.zeros_like(frame, dtype = np.float32)
        frame_avg, frame_count, frame = average_frames( vid, frame_avg, p.step,
                                                        p.alpha, frame_count)    
        # Break out of this loop if emptied stack      
        if frame is None: print ("End of stream"); break;

        if remove_bg:
            frame_avg = subtract_background(frame_avg, background)
            
        if double_substract_bg:
            frame_avg, frame_binary, cnts = subtract_stationary_background(fgbg,                                frame_avg, frame_count, **params)
        else:
            cnts = []
            frame_binary = np.empty(0)
            # cv2.normalize(  src = frame_avg, dst = frame_avg,
                            # alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX)
                  
        prob_mask= prob_mask_hsv(frame_avg, roi_hist, h_ranges,
                                                chs, **params)
        
        
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
            cent = utils.rectangle_to_centroid(pts)
            
        centroids.append(cent)
        time = float(frame_count)/fps
        times.append(time) #Save time
        if frame_binary.size:
            num_nonzero = np.count_nonzero(frame_binary)
            num_nonzeros.append(num_nonzero)

        if frame_count < 10: # requires pause for rendering, may be machine dependent -.-
            cv2.waitKey(np.int(1/fps * p.step * 1000 * 20))
        else: # requires pause for rendering, may be machine dependent -.-
            cv2.waitKey(np.int(1/fps * p.step * 1000 * 0.1))
        # Visualize
        if p.show_frames:
            # frame_vis = prob_mask.copy() # frame_avg.copy()
            frame_vis = cv2.cvtColor(frame_avg, cv2.COLOR_BGR2GRAY).astype(np.uint8)
            
            (r, b, w) = ((0, 0, 255), (0, 0, 0), (255, 255, 255))
            # Put timestamp on the average image
            time_str = "{:.2f}".format(time)
            dimy, dimx = frame_vis.shape[:2]
            time_loc = (int(dimx-250), dimy)
            cv2.putText(frame_vis, time_str, time_loc, cv2.FONT_HERSHEY_PLAIN, 5, w)
            prob_str = "Max prob: {:.2f}%".format(np.max(prob_mask) / 2.55)
            prob_loc = (50, 50)
            cv2.putText(frame_vis, prob_str, prob_loc, cv2.FONT_HERSHEY_PLAIN, 3, w)
            cv2.polylines(frame_vis, pts =[pts], isClosed= True, color= w, thickness= 2)
            # Draw location of center of mass on the average image
            cv2.circle(frame_vis, tuple(cent)[:2], radius = 4, color = b, thickness = 4)
            
            # Image should be uint8 to be drawable in 0, 255 scale
            # https://stackoverflow.com/questions/9588719/opencv-double-mat-shows-up-as-all-white
            (frame_vis, _) = utils.resize_frame(frame_vis, height = p.height_resize)
            
            cv2.imshow("Tracking", frame_vis)
            
            if frame_count > 0 and p.save_frames:
                new_shape = frame_vis.shape + (1, )
                frame_vis = np.reshape(frame_vis, new_shape)
                frame_vis = np.repeat(frame_vis,3,  axis = 2)
                vid_out.write(frame_vis)

        if p.annotate_mask and p.plot_mask and (cnts):

            w = (255, 255, 255)
            cnt_metric = cv2.arcLength(cnts[0], True)
            ann_str = "Max perim: {:.2f}, #Contours: {}, #Nonzero: {}"\
                        .format(cnt_metric, len(cnts), num_nonzero)
            ann_loc = (50, 50)
            cv2.putText(frame_binary, ann_str, ann_loc, cv2.FONT_HERSHEY_PLAIN, 3, w)
            
        if p.plot_mask and frame_binary.size: 
            (frame_binary, _) = utils.resize_frame(frame_binary, height = p.height_resize)
            cv2.imshow("Mask", frame_binary)
                    
            # Interrupt on ESC
            ch = 0xFF & cv2.waitKey(1)
            if ch == 27: break;
            elif ch == ord('d'): import pdb; pdb.set_trace()
    # end while True:
    output_data(centroids, times, num_nonzeros, video_src)
    if p.save_frames:
        vid_out.release()
    cv2.destroyAllWindows()

def main(video_src, init_flag):
    # Read video file path from user input    
    # Show documentation
    # print(__doc__)
    
    # cwd = os.getcwd()
    # (head, video_src) = os.path.split(fpath)
    # os.chdir(head)
    try:
        track_motion(video_src, init_flag)
    finally:
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main(video_src, init_flag)
