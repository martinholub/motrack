# Import packages
from scipy.spatial import distance as dist
import numpy as np
import cv2
import matplotlib.pyplot as plt
import imutils
from skimage import exposure
import sys
import pdb

# Get video:
# video_src = "videos/seagull.MP4" # for debugging, works nicely

# GoPro . MP4 Videos require specific codec -.-
# https://github.com/adaptlearning/adapt_authoring/wiki/Installing-FFmpeg
# Unfortunately it still doesnt work!!

# White table, fits in the frame
video_src = "//DCPHARMAIN/RawDataEIN/Ladina/Behaviour/Barnes maze/Cx30u43 cKO/cohort1u2/9mth/day4/GOPR0982.MP4"
# video_src = "videos/GOPR0950.MP4"

# Brown table, almost fits
#video_src = "//DCPHARMAIN/RawDataEIN/Ladina/Behaviour/Barnes maze/Cx30u43 cKO/cohort1u2/2mth/2mth-testday9/Clip181.m4v"

def startdebug():
	# Fire up debugger
	pdb.set_trace()

def resize_frame(image):
	# Read Image and force size. Save scaling ratio
	hght = 500
	ratio = image.shape[0] / hght
	image = imutils.resize(image, height = hght)
	return image, ration

def order_points(pts, ratio = 1):
	'''
	Function to put points of bounding box in clockwise order
	'''
	# sort the points based on their x-coordinates
	xSorted = pts[np.argsort(pts[:, 0]), :]
	
	# grab the left-most and right-most points from the sorted
	# x-roodinate points
	leftMost = xSorted[:2, :]
	rightMost = xSorted[2:, :]
	
	# now, sort the left-most coordinates according to their
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
	return np.array([tl, tr, br, bl], dtype="float32") * ratio


def selectROI(img, win_name):
	print("SelectROI (Enter=confirm, Esc=exit)")
	# Prompt for ROI to be analyzed
	try:
		(c, r, w, h) = cv2.selectROI(win_name, img, fromCenter = False, showCrosshair = False)
		# Esc kills the window
		if cv2.waitKey(0) & 0xff == 27:
			cv2.destroyAllWindows()
	finally:
		cv2.destroyAllWindows()
	# Store bounding box corners	
	pts = [[r, c], [r, c+w], [r+h, c], [r+h, c+w]]
	pts = order_points(np.asarray(pts))
	# Pull out roi
	roi = img[r:r+h, c: c+w]
	return pts, roi


def getGoodROI(video_src, release = False):
	# Prompt user to select a good frame we will work on later
	print ("Press key `p` to pause the video and select ROI")
	# Capture video
	vid = cv2.VideoCapture(video_src)
	try:
		while vid.isOpened():
			startdebug()
			print(vid.get(cv2.CAP_PROP_POS_FRAMES)) # debug
			# print(vid.get(cv2.CAP_PROP_FOURCC)) # debug
			# Read next frame
			retval, frame = vid.read()
			# Exit when video cant be read
			if not retval: print ('Cannot read video file'); sys.exit();
			# Pause on 'p'
			if(cv2.waitKey(10) == ord('p')):
				#Define an initial bounding box
				# bbox = (158, 26, 161, 163)
				# Uncomment the line below to select a different bounding box
				pts, roi = selectROI(frame, win_name = "SelectROI")
				break # Break from while loop

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
		
	return pts, roi, vid, frame_pos

def goToFrame(vid, frame_pos):
	# Later may add a check if video is already opened
	vid.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
	return vid
	
def main():
	pts, roi, vid, frame_pos = getGoodROI(video_src)
#######################################################
if __name__ == '__main__':
	main()