'''
Lucas-Kanade tracker
====================

Lucas-Kanade sparse optical flow demo. Uses goodFeaturesToTrack
for track initialization and back-tracking for match verification
between frames.

Usage
-----
lk_track.py [<video_source>]


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
step = 3
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
	
	# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
	term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
	
	# Get FPS
	fps = vid.get(cv2.CAP_PROP_FPS)
	frame_count = 0
	
	
	while True:
		frame_avg = np.zeros_like(frame[:,:,0], dtype = np.float32)
		frame_count += 1
		for i in range(step):
			
			ret, frame = vid.read()
			if frame is None: break;
			frame = frame.astype(np.uint8)
			
			frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			frame_gray = cv2.GaussianBlur(frame_gray, kSize, sigmaX)
			
			cv2.accumulateWeighted(frame_gray, frame_avg, alpha)
			
			# Get current time
			time = float(frame_count)/fps
		
		if frame is None: print ("End of stream"); break;	
		frame_roi = frame_avg[int(bbox[1] - pad) : int(bbox[1] + bbox[3] + pad),
							int(bbox[0] - pad) : int(bbox[0] + bbox[2] + pad)]
		# frame_roi_filt = gaussian(frame_roi, sigma = sigma, multichannel = False)
		
		
		thresh = threshold_otsu(frame_roi);
		binary = frame_roi < thresh
		selem = diamond(10)
		binary_cl = binary_closing(binary, selem)
		
		mask = np.zeros_like(frame_avg)
		mask[int(bbox[1] - pad) : int(bbox[1] + bbox[3] + pad), int(bbox[0] - pad) : int(bbox[0] + bbox[2] + pad)] = binary_cl
		mask = mask.astype(np.uint8)
		
		binary_label = label(mask)
		binary_props = regionprops(binary_label)
		cent=(int(binary_props[0].centroid[0]), int(binary_props[0].centroid[1]))
		
		
		bboxRP = binary_props[0].bbox
		
		bbox = (bboxRP[1], bboxRP[0], int(bboxRP[3] - bboxRP[1]), int(bboxRP[2] - bboxRP[0]))
		
		# startdebug()
		# debugPlot(frame_avg, bbox, mask, frame_roi, binary_cl)
		
		# apply meanshift to get the new location
		ret, bbox = cv2.CamShift(mask, bbox, term_crit)
		red = (0, 0, 255)
		cv2.circle(frame_avg, cent, radius = 3, color = red)
		time_str = "{:.2f}".format(time)
		dimx, _ = frame_avg.shape
		time_loc = (int(dimx-50), 50)
		black = (0, 0, 0)
		cv2.putText(frame_avg, time_str, time_loc, cv2.FONT_HERSHEY_PLAIN, 4, black)
		
		p1 = (int(bbox[0]), int(bbox[1]))
		p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
		cv2.rectangle(frame_avg, p1, p2, (0,0,255))
		# roi_hist = cv2.calcHist(images = [frame_avg],
								# channels = [0],
								# mask = mask,
								# histSize = [180],
								# ranges = [0, 180],
								# uniform = True
								# )
		# cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

		cv2.imshow('Tracking', frame_avg)

		ch = 0xFF & cv2.waitKey(1)
		if ch == 27: break; 

def main():
	try: 
		video_src = ap.parse_args().video
	except: 
		video_src = 0

	print(__doc__)
	
	try:
		motracker(video_src)
	finally:
		cv2.destroyAllWindows()				

if __name__ == '__main__':
	main()