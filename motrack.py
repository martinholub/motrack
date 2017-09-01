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

##############################################################
## Parse arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type = str, help= "path to the video file")
##############################################################
## Define global variables
alpha = 0.5
step = 10
sigmaX = 1
kSize = (3, 3)
pad = 10

#############################################################
############## FUNCTION DEFINITIONS #########################
#############################################################

def startdebug():
	# Fire up debugger
	pdb.set_trace()
############################################################
def debugPlot(im, bbox, mask,  roi, roi_mask):
	plt.close()
	fig, ax = plt.subplots(2, 2, figsize = (10, 8))
	ax[0, 0].imshow(im, cmap = plt.cm.gray)
	pat = rect((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth = 2, edgecolor = 'r', facecolor = 'none')
	ax[0, 0].add_patch(pat)
	
	pat2 = rect((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth = 2, edgecolor = 'r', facecolor = 'none')
	ax[0, 1].imshow(mask, cmap = plt.cm.gray)
	ax[0, 1].add_patch(pat2)
	
	ax[1, 0].imshow(roi, cmap = plt.cm.gray)
	ax[1, 1].imshow(roi_mask, cmap = plt.cm.gray)
	plt.show()
############################################################
def labelthresh(frame_avg, bbox, pad):
	'''
	Here we try to facilitate object tracking by creating first thresholding in roi and then creating a binary mask. This may not be necessary and may not work.
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
	#############################################################
def getroihist(frame, bbox, pad):
	# Setup ROI for tracking. We will be basically looking maximum correlation of HSV histogram of the roi (our template of animal) across subsequent frames.
	# Pad the bbox for added robustness (?)
	# Mind the order! track_window = (c,r,w,h) -> roi = frame[r:r+h, c:c+w]
	frame_roi = frame[int(bbox[1] - pad) : int(bbox[1] + bbox[3] + pad),
						int(bbox[0] - pad) : int(bbox[0] + bbox[2] + pad)]
	hsv_roi = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	# Threshold on color, here you must introduce some a priori knowledge
	mask = cv2.inRange(hsv_roi,np.array((0., 60.,32.)),np.array((180.,255.,255.)))
	# Get roi histogram. Dunno why exactly in these ranges though!
	roi_hist = cv2.calcHist(images = [hsv_roi], channels = [0], mask = mask, histSize = [180], ranges = [0, 180])
	# Normalize in place between alpha and beta
	cv2.normalize(roi_hist, roi_hist, alpha=0, beta=255, norm_type = cv2.NORM_MINMAX)
	return roi_hist
	############################################################
def getprobmask_hsv(frame_avg, roi_hist):

	frame_avg_hsv = cv2.cvtColor(frame_avg, cv2.COLOR_BGR2HSV)
	prob_mask = cv2.calcBackProject(images = [frame_avg_hsv], channels = [0], hist = roi_hist, ranges = [0, 180], scale = 1) 
	return prob_mask
	############################################################
def evaldist(pts):
	Xs = [p[0] for p in pts]
	Ys = [p[1] for p in pts]
	cX = np.mean(Xs, dtype = np.uint8)
	cY = np.mean(Ys, dtype = np.uint8)
	
	
	# cX = 
	pass
	############################################################

def motracker(video_src):
	# Capture video
	vid = cv2.VideoCapture(video_src)
	# Read first frame.
	ok, frame = vid.read()
	if not ok: print ('Cannot read video file'); sys.exit();
	
	#Define an initial bounding box
	# bbox = (126, 23, 188, 164)
	# Uncomment the line below to select a different bounding box
	bbox = cv2.selectROI("Tracking", frame, fromCenter = False, showCrosshair = False)
	
	roi_hist = getroihist(frame, bbox, pad)
	
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
			frame = frame.astype(np.uint8) # convert to float for opencv
			# Accumulate running average
			cv2.accumulateWeighted(frame, frame_avg, alpha)
				
		# Break also out of this loop if emptied stack		
		if frame is None: print ("End of stream"); break;
		# Get current time
		time = float(frame_count)/fps
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
		prob_mask = getprobmask_hsv(frame_avg, roi_hist)
		##########################################################
		
		# Apply meanshift to get the new location
		ret, bbox = cv2.CamShift(prob_mask, bbox, term_crit)
		
		# Define colors
		red = (0, 0, 255)
		black = (0, 0, 0)
		# Draw location of center of mass on the average image
		# cv2.circle(frame_avg, cent, radius = 3, color = red)
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
		startdebug()
		frame_avg = cv2.polylines(img = frame_avg, pts = [pts],isClosed = True, color = red, thickness = 2)
		
		# Plot all the average frame
		# Image should be uint8 to be drawable in 0, 255 scale
		# https://stackoverflow.com/questions/9588719/opencv-double-mat-shows-up-as-all-white
		cv2.imshow('Tracking', frame_avg)
		
		# Next you need to:
		# - save current position of bbox centroid and evaluate distance traveled
		# - save the elapsed time
		# - Make the search more robust (try different hardcoded values, some more filtering, ...)
		# Have one function for the same part of algorithm, but pass different flags to have different aproaches
		
		# Interrup on ESC
		ch = 0xFF & cv2.waitKey(60)
		if ch == 27: break;
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
	