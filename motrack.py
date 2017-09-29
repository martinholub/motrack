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
from time import clock
from skimage.filters import gaussian, threshold_otsu
from skimage.morphology import binary_closing, diamond, disk, square
from skimage.measure import label, regionprops
import argparse
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle as rect
import pdb
import getcoords

# Parse arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type = str, help= "path to the video file")

## Define global variables
alpha = 0.5 # Weighting for running sums
step = 5 # Number of steps accumulated in running sum
sigmaX = 1 # std of Gaussian kernel
kSize_gauss = (3, 3) # Gaussian kernel size
kSize_canny = (5, 5)
pad = 10 # optional padding of a roi

def startdebug():
    """Start debugger here
    """
    pdb.set_trace()

def debug_plot(im, bbox, mask,  roi, roi_mask):
    """Helper debugging plot
    """
    fig, ax = plt.subplots(2, 2, figsize = (10, 8))
    ax[0, 0].imshow(im, cmap = plt.cm.gray)
    pat = rect((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth = 2,
                edgecolor = 'r', facecolor = 'none')
    ax[0, 0].add_patch(pat)
    
    pat2 = rect((   bbox[0], bbox[1]), bbox[2], bbox[3], linewidth = 2, 
                    edgecolor = 'r', facecolor = 'none')
    ax[0, 1].imshow(mask, cmap = plt.cm.gray)
    ax[0, 1].add_patch(pat2)
    
    ax[1, 0].imshow(roi, cmap = plt.cm.gray)
    ax[1, 1].imshow(roi_mask, cmap = plt.cm.gray)
    plt.show()

def debug_plot2(frame, pts, roi):
    """Helper debugging plot
    """
    fig, (ax1, ax2) = plt.subplots(1,2, figsize = (12,8))
    if type(pts) is tuple:
        pat = rect((pts[0], pts[1]), pts[2], pts[3], linewidth = 2,
                edgecolor = 'r', facecolor = 'none')
        ax1.add_patch(pat)
    else:
        ax1.scatter(pts[:,0], pts[:,1], s = 100, c = "c",marker = "o")
    ax1.imshow(frame, cmap = "gray")
    ax2.imshow(roi, cmap = "gray")
    plt.show()

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
    
    (hsv_lowerb, hsv_upperb) = select_hsv_range(vid, video_src, background,
                                                frame_pos, reinitialize)
    chs = [1, 2]
    h_sizes = [256, 256]
    h_ranges = [0, 256, 0, 256]
    
    roi_hsv = cv2.cvtColor(roi_rgb, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(roi_hsv, hsv_lowerb , hsv_upperb)
    roi_hist = cv2.calcHist(images = [roi_hsv], channels = chs, mask = mask,
                            histSize = h_sizes, ranges = h_ranges)                        
    # Normalize in-place between alpha and beta
    cv2.normalize(  src = roi_hist, dst = roi_hist,  alpha = 0, beta = 255,
                    norm_type = cv2.NORM_MINMAX)
    return roi_hist, h_sizes, h_ranges, chs

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
                    
        frame, _ = getcoords.resize_frame(frame, height = 500)
                
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
    
def prob_mask_hsv(frame_avg, roi_hist, h_ranges, chs):
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
    """
    frame_avg_hsv = cv2.cvtColor(frame_avg, cv2.COLOR_BGR2HSV)
    prob_mask = cv2.calcBackProject(images = [frame_avg_hsv], channels = chs, 
                                    hist = roi_hist, ranges =h_ranges, scale= 1) 
    return prob_mask

def output_data(centroids, times):
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
    
    with open("data.txt", "w") as f:
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
    
def bbox_to_pts(bbox):
    """Converts tuple of 4 values to bbox vertices 
    """
    pts = np.array([[ bbox[0], bbox[1]], #[tl, tr, br, bl]
                    [ bbox[0]+ bbox[2], bbox[1]],
                    [ bbox[0]+ bbox[2], bbox[1]+ bbox[3]],
                    [ bbox[0], bbox[1]+ bbox[3]]], dtype = np.int32) 
    pts = getcoords.order_points(pts)
    return pts

def pts_to_bbox(pts):
    """Converts 4 x,y coordinate pairs to corresponding bbox tuple
    """
    pts = getcoords.order_points(pts)
    bbox = (pts[0], pts[1], pts[2] - pts[0], pts[3] - pts[1])
    return bbox

def tracker(prob_mask, bbox, type = "meanShift", **kwargs):
    """Tracks an object in image
    
    The function is to be extended at will to achieve reliable tracking of given
    object.
    
    Parameters
    ---------------
    type : str
        Defines the tracking method to use
    
    Returns
    -----------
    bbox : ndarray
    pts : ndarray
    """
    bbox_old = bbox
    pts_old = bbox_to_pts(bbox_old)
    
    if type == "meanShift":
        ret, bbox = cv2.meanShift(prob_mask, bbox, kwargs["term_crit"] )
        pts = bbox_to_pts(bbox)
        
    elif type == "CamShift":    
        ret, bbox = cv2.CamShift(prob_mask, bbox, kwargs["term_crit"])
        pts = cv2.boxPoints(ret)
        
    elif type == "threshold":
        try:
            (frame_vis, props) =    label_contours(prob_mask, 
                                    kwargs["kSize_gauss"], kwargs["sigmaX"],
                                    kwargs["kSize_canny"])
        except IndexError:
            return None
            
        bbox = (props.bbox[1], props.bbox[0],
                np.int(props.bbox[3]-props.bbox[1]), np.int(props.bbox[2] - props.bbox[0]))
        pts = bbox_to_pts(bbox)
        centroid = props.centroid
    
    c_old = rectangle_to_centroid(pts_old)
    c = rectangle_to_centroid(pts)
    dist = np.sqrt((c[0] - c_old[0])**2 + (c[1] - c_old[1])**2)
    
    if dist < 20:
        bbox = bbox_old
        pts = pts_old
    
    return (bbox, pts)

def label_contours(prob_mask, kSize_gauss, sigmaX, kSize_canny, type = 3):
    """Find binary image of object of interest
    
    Facilitate object tracking by first thresholding roi and then creating
    a binary mask
    
    Parameters
    ----------
    pad : int
        Optinally specifies number of pixel to be added on each side of 
        the bounding box
    bbox : tuple
        (minY, minX, deltaY, deltaX)
    Returns
    ----------
    
    """
    prob_mask = cv2.GaussianBlur(prob_mask, kSize_gauss, sigmaX)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kSize_canny)
    vis_img = np.zeros_like(prob_mask, dtype = np.uint8)
    
    if type == 1:
        edged = cv2.Canny(prob_mask, threshold1 = 120, threshold2 = 255)
        edged = cv2.dilate(edged, kernel = kernel, iterations = 2)
        
        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)[1]
        cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]
        
        cv2.drawContours(vis_img, cnts, contourIdx = -1, 
                        color = (255, 255, 255), thickness =  -1)
        # vis_img = cv2.morphologyEx(vis_img, cv2.MORPH_CLOSE, kernel, iterations =1)
        vis_img = cv2.dilate(vis_img, kernel = kernel, iterations = 5)
        
    elif type == 2:
        binary = np.where(prob_mask > np.max(prob_mask) * 0.25, 255, 0).astype(np.uint8)
        binary = cv2.erode(binary, kernel = kernel, iterations = 1)
        binary = cv2.dilate(binary, kernel = kernel, iterations = 5)
        #binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations =1)

        cnts = cv2.findContours(binary.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)[1]       
        #cnts = [cnt for cnt in cnts if cv2.contourArea(cnt) > 1500]
        cnts = sorted(cnts, key = cv2.contourArea, reverse = False)[:5]
        vis_img = vis_img.astype(np.float32)
        cv2.drawContours(vis_img, cnts, contourIdx = -1, 
                        color = (255, 255, 255), thickness =  -1)
        # vis_img = binary.copy()
        # cnts = []
                        
    elif type == 3:
        from skimage import measure
        labels = measure.label(binary)
        props = measure.regionprops(labels)
        props = sorted(props, key = lambda x: x.area , reverse = False)[:1]
        
        for prop in props:
            try:
                label_mask = np.where(labels == prop.label, 255, 0).astype(np.uint8)
                vis_img = cv2.add(vis_img, label_mask)
            except IndexError:
                pass
        
        labels = measure.label(vis_img)
        cnts = measure.regionprops(labels)[0]
    elif type == 4:
        # indices = np.where(prob_mask > np.max(prob_mask) * 0.5)
        # from sklearn.metrics.pairwise import pairwise_distances
        # dist_mat = pairwise_distances(indices, metric = "euclidean")
        # from sklearn.cluster import DBSCAN
        # db = DBSCAN(eps = .3, min_samples = 2).fit(dist_mat)    
        #
        pass
    return (vis_img, cnts)

def segment_background(frame, pts):
    """Substract stationary background
    
    """
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(frame_gray)
    bbox = cv2.boundingRect(pts)
    pts = swap_coords_2d(pts)
    y_pad = 140 # cover also mouse tail
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
    
def track_motion(video_src, remove_bg = False, reinitialize = False):

    if reinitialize:
        pts, roi, vid, frame_pos, frame = getcoords.select_roi_video(video_src)
    else:
        pts, roi, vid, frame_pos, frame = from_preset(video_src)
        
    # Convert between different rectangle representations
    pts = swap_coords_2d(pts)
    bbox = cv2.boundingRect(pts)
    
    if remove_bg : 
        background, mask_fgd = segment_background(frame, pts)
        frame_bg_removed = subtract_background(frame, background)
        roi = frame_bg_removed[pts[0][0]:pts[1][0], pts[0][1]:pts[2][1],:]
    else :
        background = np.empty(0);
    
    roi_hist, sizes, ranges, chs = get_roi_hist(roi, vid, background, frame_pos,
                                                reinitialize)
    # Setup the termination criteria, either 10 iteration or move by atleast 1 pt
    term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
    
    fps = vid.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    # Lists to temporalily store output data
    times = []
    centroids = []    
    while vid.isOpened():
        frame_avg = np.zeros_like(frame, dtype = np.float32)
        frame_avg, frame_count, frame = average_frames( vid, frame_avg, step,
                                                        alpha, frame_count)    
        # Break out of this loop if emptied stack      
        if frame is None: print ("End of stream"); break;
                
        if remove_bg:
            frame_avg = subtract_background(frame_avg, background)
        
        prob_mask = prob_mask_hsv(frame_avg, roi_hist, ranges, chs)
        #(prob_mask, _) = label_contours(prob_mask, kSize_gauss, sigmaX, 
                                        #kSize_canny, type = 2)
        params_tracker = dict(  kSize_gauss = kSize_gauss, sigmaX = sigmaX,
                        kSize_canny = kSize_canny, term_crit = term_crit)
        res = tracker(  prob_mask, bbox, "meanShift", **params_tracker)
        
        if res is None:
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
        (frame_vis, _) = getcoords.resize_frame(frame_vis, height = 500)
        cv2.imshow("Tracking", frame_vis)
                
        # Interrup on ESC
        ch = 0xFF & cv2.waitKey(60)
        if ch == 27: break;
    # end while True:
    output_data(centroids, times)
    cv2.destroyAllWindows()

def main():
    # Read video file path from user input
    try: 
        video_src = ap.parse_args().video
    except: 
        video_src = 0
    # Show documentation
    print(__doc__)
    # Assure that all windows get destroyed at the end of the run
    try:
        track_motion(video_src)
    finally:
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
