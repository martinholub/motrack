import numpy as np
import cv2
import argparse
import pdb
import getcoords
import params as p
import motrack
import glob

basepath = "Q:\EIN Group\MartinHolub\1mth-day1"
file_name = "*.m4v"
base_skip = 400
roi_pad = 250
win_name = "Press 'Enter' to select frame"
initialize = True
    
    
def find_frame_video(   video_src, show_frames = False, release = False):
    """Select a frame from arbitrary video.
    
    Prompts the user to select a frame in video.
    
    Parameters
    ----------
    video_src : str
        path to video (GOPRO videos not yet working!)
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
        
    References:
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
            if(cv2.waitKey(10) == ord('p')):
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

for i, file in enumerate(list(glob.glob(basepath + file_name))):
    # open video
    # skip x initial frames
    # get frame rate
    # get number of frames in video
    pass
    
