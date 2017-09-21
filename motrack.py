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
import sys
import pdb
import getcoords

## Parse arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type = str, help= "path to the video file")

## Define global variables
alpha = 0.5
step = 5
sigmaX = 1
kSize = (3, 3)
pad = 10
boxList = []
timeList = []
centList = []

def startdebug():
    # Fire up debugger
    pdb.set_trace()

def debug_plot(im, bbox, mask,  roi, roi_mask):
    plt.close()
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

def label_thresh(frame_avg, bbox, pad):
    '''
    Here we try to facilitate object tracking by creating first thresholding 
    in roi and then creating a binary mask. This may not be necessary and may not work.
    '''
    # Convert frame to grayscale
    frame_avg_gray = cv2.cvtColor(frame_avg, cv2.COLOR_BGR2GRAY)
    # Blur frame with Gaussian
    frame_avg_gray = cv2.GaussianBlur(frame_avg_gray, kSize, sigmaX)
    # Pull out only subregion
    # Pad the frame to capture the whole object
    frame_roi = frame_avg_gray[int(bbox[1] - pad) : int(bbox[1] + bbox[3] + pad),
                        int(bbox[0] - pad) : int(bbox[0] + bbox[2] + pad)]
    # frame_roi_filt = gaussian(frame_roi, sigma = sigma, multichannel = False)
    
    # Aply Automatic threshold
    thresh = threshold_otsu(frame_roi);
    binary = frame_roi < thresh # May need to change sign for different img
    # Run closing on binary image with diamond kernel
    selem = diamond(10) 
    binary_cl = binary_closing(binary, selem)
    # Create mask where we will put our thresholded object
    mask = np.zeros_like(frame_avg_gray, dtype = np.uint8)
    # Put the boundig box into the mask
    mask[int(bbox[1] - pad) : int(bbox[1] + bbox[3] + pad), int(bbox[0] - pad) : int(bbox[0] + bbox[2] + pad)] = binary_cl
    # mask = mask.astype(np.uint8)
    
    # Label connected components
    binary_label = label(mask)
    # Extract location of centroid and bbox of the thresholded object
    binary_props = regionprops(binary_label)
    cent=(int(binary_props[0].centroid[0]), int(binary_props[0].centroid[1]))       
    bboxRP = binary_props[0].bbox
    # We need to shuffle the dimensions quite a bit
    bbox = (bboxRP[1], bboxRP[0], int(bboxRP[3] - bboxRP[1]), int(bboxRP[2] - bboxRP[0]))
    return bbox, mask, frame_avg_gray

def get_roi_hist(frame, roi_rgb, video_src, vid = [], frame_pos = []):
    # Setup ROI for tracking. We will be basically looking maximum correlation of HSV histogram of the roi (our template of animal) across subsequent frames.
    # Pad the bbox for added robustness (?)
    # Mind the order! track_window = (c,r,w,h) -> roi = frame[r:r+h, c:c+w]
    
    # need to pick position from video and than display hsv slider for boundary selection
    
    (hsv_lowerb, hsv_upperb) = select_hsv_range(vid, video_src, frame_pos)
    s_size = hsv_upperb[1] - hsv_lowerb[1]
    v_size = hsv_upperb[1] - hsv_lowerb[2]
    s_range = np.arange(hsv_lowerb[1], hsv_upperb[1] + 1)
    v_range = np.arange(hsv_lowerb[2], hsv_upperb[2] + 1)
    chs = [1, 2]
    
    roi_hsv = cv2.cvtColor(roi_rgb, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(roi_hsv, hsv_lowerb , hsv_upperb)
    # Returned null without setting an error! - what is wrong?
    roi_hist = cv2.calcHist(images = [roi_hsv], channels = chs, mask = mask,
                            histSize = [s_size, v_size], ranges = [s_range, v_range])
                            
    # Normalize in-place between alpha and beta
    cv2.normalize(src = roi_hist,dst = roi_hist,  alpha=0, beta=255, norm_type = cv2.NORM_MINMAX)
    return roi_hist, [s_size, v_size], [s_range, v_range], chs

def select_hsv_range(vid, video_source, frame_pos = [], change_flag = False):
    if change_flag:
        (_, frame) = getcoords.go_to_frame( vid, frame_pos, video_source, 
                                            return_frame =True)
        frame, _ = getcoords.resize_frame(frame, height = 500)
        
        def nothing(x):
            pass
        
        hh='Hue High'
        hl='Hue Low'
        sh='Saturation High'
        sl='Saturation Low'
        vh='Value High'
        vl='Value Low'
        cv2.namedWindow("Select HSV Range")
        cv2.resizeWindow('Select HSV Range', frame.shape[1], frame.shape[0] )
        print("Change ranges on sliders and press Enter to update")

        cv2.createTrackbar(hl, 'Select HSV Range',0,179,nothing) # ~180 deg
        cv2.createTrackbar(hh, 'Select HSV Range',0,179,nothing)
        cv2.createTrackbar(sl, 'Select HSV Range',0,255,nothing)
        cv2.createTrackbar(sh, 'Select HSV Range',0,255,nothing)
        cv2.createTrackbar(vl, 'Select HSV Range',0,255,nothing)
        cv2.createTrackbar(vh, 'Select HSV Range',0,255,nothing)

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
    else:
        hsv_lowerb = np.array([0, 50, 0])
        hsv_upperb = np.array([179, 160, 100])
        
    return (hsv_lowerb, hsv_upperb)
    
def prob_mask_hsv(frame_avg, roi_hist, ranges, sizes, chs):

    frame_avg_hsv = cv2.cvtColor(frame_avg, cv2.COLOR_BGR2HSV)
    prob_mask = cv2.calcBackProject(images = [frame_avg_hsv], channels = chs, 
                                    hist = roi_hist, ranges = ranges, scale = 1) 
    return prob_mask
    ############################################################

def evaldist(centList, timeList):
    '''
    '''
    # Initalize variables that hold sums
    totDist = 0
    # Open file to save postprocessed data
    with open("data.txt", "w") as f:
        f.write("No.,cX,cY,time,dist\n") # Write header
        for i, cent in enumerate(centList):
            # Pull out (X, Y) coords
            cX = cent[0]
            cY = cent[1]
            time = timeList[i] # Pull out time 
            if i > 0 : # Compute distance traveled
                dx = (cX - centList[i-1][0])
                dy = (cY - centList[i-1][1])
                dist = np.sqrt(dx**2 + dy**2)
            else: 
                dist = 0;   # We dont have information on previous location
            # dump to file
            f.write("{:d},{:0.2f},{:0.2f},{:0.2f},{:0.2f}\n".format(i, cX, cY, time, dist))
            # Incement distance tracker
            totDist += dist # in pixels - will need to calibrate??
        # Wite total distance and time to file
        f.write("Total dist in pix: {:0.4f}\n".format(totDist))
        f.write("Total timein sec: {:0.4f}\n".format(timeList[-1]))
    # f.close() is implicit
    ############################################################
def getcent(pts):
    # Pull out vertex coordinates
    Xs = [p[0] for p in pts]
    Ys = [p[1] for p in pts]
    # Get mean coordinates in X and Y -> centroid of bbox
    cX = np.mean(Xs, dtype = np.float32)
    cY = np.mean(Ys, dtype = np.float32)
    # Append coordinates to list
    return (cX, cY)
    
def motracker(video_src):

    pts, roi, vid, frame_pos, frame = getcoords.select_roi_video(video_src)
    
    roi_hist, sizes, ranges, chs = get_roi_hist(frame, roi, video_src, vid, frame_pos)
    
    # Setup the termination criteria, either 10 iteration or move by atleast 1 pt
    term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
    
    # Get FPS
    fps = vid.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    
    while True:
        # Create aray for storing average image
        frame_avg = np.zeros_like(frame, dtype = np.float32)
        frame_count += 1 # Increment count for timing
        
        for i in range(step):
            # Running average
            ret, frame = vid.read() # read next frame 
            if frame is None: break; # if stack empty, break
            frame = frame.astype(np.uint8) # assure format, uint8
            # Accumulate running average
            cv2.accumulateWeighted(frame, frame_avg, alpha)
                
        # Break also out of this loop if emptied stack      
        if frame is None: print ("End of stream"); break;
        # Get current time
        time = float(frame_count)/fps
        timeList.append(time) #Save time 
        # Normalize pixel values between 0 and 255
        frame_avg = cv2.normalize(frame_avg, frame_avg, alpha=0, beta=255,
        norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        ############################################################    
        # Run function that creates binary mask from bbox. 
        # bbox, prob_mask, frame_avg = labelthresh(frame_avg, bbox, pad)
        ###########################################################
        # startdebug()
        # debugPlot(frame_avg, bbox, mask, frame_roi, binary_cl)
        ##########################################################
        # Run function that gives probability mask from hsv histogram
        prob_mask = prob_mask_hsv(frame_avg, roi_hist, ranges, sizes, chs)
        ##########################################################
        
        # Apply meanshift to get the new location
        ret, bbox = cv2.CamShift(prob_mask, bbox, term_crit)
        
        # Define colors
        red = (0, 0, 255)
        black = (0, 0, 0)
        # Put timestamp on the average image
        time_str = "{:.2f}".format(time)
        dimy, dimx, _ = frame_avg.shape
        time_loc = (int(dimx-50), dimy-50)
        frame_avg = cv2.putText(frame_avg, time_str, time_loc, cv2.FONT_HERSHEY_PLAIN, 1, black)
        # Draw bbox on the average image
        # p1 = (int(bbox[0]), int(bbox[1]))
        # p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        # frame_avg = cv2.rectangle(frame_avg, p1, p2, color = red)
        # Different way how to draw a box
        pts = cv2.boxPoints(ret)
        pts = np.int64(pts)
        frame_avg = cv2.polylines(img = frame_avg, pts = [pts],isClosed = True, color = red, thickness = 2)
        # Get centroid of bbox, append to List
        cent = getcent(pts)
        centList.append(cent)
        # Draw location of center of mass on the average image
        cv2.circle(frame_avg, cent, radius = 4, color = red, thickness = 4)
        
        # Plot all the average frame
        # Image should be uint8 to be drawable in 0, 255 scale
        # https://stackoverflow.com/questions/9588719/opencv-double-mat-shows-up-as-all-white
        cv2.imshow('Tracking', frame_avg)
                
        # Interrup on ESC
        ch = 0xFF & cv2.waitKey(60)
        if ch == 27: break;
    ############################################################ end while True:
    # Run postprocessing functions
    evaldist(centList, timeList)
    # release video object
    vid.release()

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
# Next you need to:
# - save current position of bbox centroid and evaluate distance traveled = OK
# - save the elapsed time = OK
# - Create projection matrix and use it to get real distance traveled
# - Make the search more robust (try different hardcoded values, some more filtering, ...)
# Have one function for the same part of algorithm, but pass different flags to have different aproaches