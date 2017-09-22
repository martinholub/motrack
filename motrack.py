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
kSize = (3, 3) # Gaussian kernel size
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

def label_thresh(frame_avg, bbox, pad):
    """Find binary image of object of interest
    
    Facilitate object tracking by first thresholding roi and then creating
    a binary mask. Currently not used.
    
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
    frame_avg_gray = cv2.cvtColor(frame_avg, cv2.COLOR_BGR2GRAY)
    frame_avg_gray = cv2.GaussianBlur(frame_avg_gray, kSize, sigmaX)
    frame_roi = frame_avg_gray[int(bbox[1] - pad) : int(bbox[1] + bbox[3] + pad),
                               int(bbox[0] - pad) : int(bbox[0] + bbox[2] + pad)]
    # Aply Automatic threshold
    thresh = threshold_otsu(frame_roi);
    binary = frame_roi < thresh # May need to change sign for different img
    selem = diamond(10) 
    binary_cl = binary_closing(binary, selem)
    # Create mask where we will put our thresholded object
    mask = np.zeros_like(frame_avg_gray, dtype = np.uint8)
    # Put the boundig box into the mask
    mask[   int(bbox[1] - pad) : int(bbox[1] + bbox[3] + pad),
            int(bbox[0] - pad) : int(bbox[0] + bbox[2] + pad)] = binary_cl
    # Label connected components
    binary_label = label(mask)
    # Extract location of centroid and bbox of the thresholded object
    binary_props = regionprops(binary_label)
    cent=(int(binary_props[0].centroid[0]), int(binary_props[0].centroid[1]))       
    bboxRP = binary_props[0].bbox
    # We need to shuffle the dimensions quite a bit
    bbox = (bboxRP[1], bboxRP[0], int(bboxRP[3] - bboxRP[1]), int(bboxRP[2] - bboxRP[0]))
    return bbox, mask, frame_avg_gray

def get_roi_hist(roi_rgb, vid, frame_pos = []):
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
    
    (hsv_lowerb, hsv_upperb) = select_hsv_range(vid, video_src, frame_pos)
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

def select_hsv_range(vid, video_source, frame_pos = [], reinitialize = False):
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
        hsv_lowerb = np.array([0, 50, 0])
        hsv_upperb = np.array([179, 160, 100])
        
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
    ############################################################

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
    
    
def motracker(video_src, reinitialize = False):

    if reinitialize:
        pts, roi, vid, frame_pos, frame = getcoords.select_roi_video(video_src)
    else:
        pts, roi, vid, frame_pos, frame = from_preset(video_src)
                     
    roi_hist, sizes, ranges, chs = get_roi_hist(roi, vid, frame_pos)

    # Convert between different rectangle representations
    pts = swap_coords_2d(pts)
    bbox = cv2.boundingRect(pts)
    
    # Setup the termination criteria, either 10 iteration or move by atleast 1 pt
    term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
    
    fps = vid.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    # Lists to temporalily store output data
    times = []
    centroids = []
    
    while vid.isOpened():
        frame_avg = np.zeros_like(frame, dtype = np.float32)
        frame_count += 1 * step
        
        for i in range(step):
            # Running average
            ret, frame = vid.read()
            if frame is None: break; # if stack empty, break
            frame = frame.astype(np.uint8)
            # Accumulate running average with alpha weighting
            cv2.accumulateWeighted(src = frame, dst = frame_avg, alpha = alpha)
            
        # Break also out of this loop if emptied stack      
        if frame is None: print ("End of stream"); break;
        
        time = float(frame_count)/fps
        times.append(time) #Save time 
        
        # Normalize pixel values between 0 and 255
        frame_avg = cv2.normalize(  frame_avg, frame_avg, alpha=0, beta=255,
                                    norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        

        prob_mask = prob_mask_hsv(frame_avg, roi_hist, ranges, chs)
        
        # Apply meanshift to update object
        ret, bbox = cv2.CamShift(prob_mask, bbox, term_crit)
        
        pts = cv2.boxPoints(ret)
        pts = np.int64(pts)
        cent = rectangle_to_centroid(pts)
        centroids.append(cent)
        
        # Visualize
        frame_vis = frame_avg.copy()
        (r, b, w) = ((0, 0, 255), (0, 0, 0), (255, 255, 255))
        
        # Put timestamp on the average image
        time_str = "{:.2f}".format(time)
        dimy, dimx, _ = frame_vis.shape
        time_loc = (int(dimx-250), dimy-150)
        cv2.putText(frame_vis, time_str, time_loc, cv2.FONT_HERSHEY_PLAIN, 5, w)
        cv2.polylines(frame_vis, pts =[pts], isClosed= True, color= r, thickness= 2)
        # Draw location of center of mass on the average image
        cv2.circle(frame_vis, tuple(cent)[:2], radius = 4, color = r, thickness = 4)
        
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
        motracker(video_src)
    finally:
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
