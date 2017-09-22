# Import packages
from scipy.spatial import distance as dist
import numpy as np
import cv2
import matplotlib.pyplot as plt
import imutils
from skimage import exposure
import sys
import pdb

def startdebug():
    # Fire up debugger
    pdb.set_trace()

def resize_frame(image, height = 860):
    '''Resize image frame.
    
    Parameters
    ------------
    image: ndaray
    
    Returns
    -----------
    array_like: resized image
    ration: scaling ratio
    '''
    ratio = image.shape[0] / height
    image = imutils.resize(image, height = height)
    return image, ratio

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
   
def fit_undistort(frame, mtx, dist, refit = False):
    '''
    Eventually this function will apply undistortion for our specific camera.
    Currently it doesn't do any good.
    
    References:http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html#undistortion
    '''   
    if refit:
        # you may want to refit the intrinsic matrix
        h, w = frame.shape[:2]
        mtx_new, roi_crop = cv2.getOptimalNewCameraMatrix(
                    mtx, dist, imageSize = (w,h), alpha = 1, newImgSize = (w,h))
        # undistort using the new found matrix
        dst = cv2.undistort(frame, mtx, dist, dst = None, newCameraMatrix = mtx_new)
        # Crop empty pixels
        x,y,w,h = roi_crop
        dst = dst[y:y+h, x:x+w]
        
    else:
        # Just apply undistortion with unchanged params
        dst = cv2.undistort(frame, mtx, dist)
        
    return dst

def select_roi(img, win_name, undistort = False):
    '''Select ROI from image.
    
    Prompts user to interactivelly select a ROI in an image.
    
    Parameters
    ------------
    img : ndarray
        frame to select ROI in
    win_name : str
        name of the plotting window
    undistort: bool
        a boolean flag indicating whether to apply undistortion
        
    Returns
    ---------
    pts : ndarray
        4x2 array of box points
    roi : ndarray
        image cropped to pts
    '''
    print("SelectROI (Enter=confirm, Esc=exit)")
    # Prompt for ROI to be analyzed
    try:
        if undistort: # currently it doesn't help at all
            pass
            # img = fit_undistort(img, intrinsic_matrix, distortion_coeffs)
        
        (c, r, w, h) = cv2.selectROI(   win_name, img, fromCenter = False,
                                        showCrosshair = False)
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

def apply_pad(image, pad, mode = 'symmetric'):
    """ Apply padding to an image
    
    Parameters
    ----------
    pad : tuple
        (y_pad, x_pad)
    """
    (y_pad, x_pad) = pad
    if len(image.shape) >= 2:
        pad_vector = [(y_pad, y_pad), (x_pad, x_pad)]
    elif len(image.shape) == 3:
        pad_vector.append((0, 0))
    image = np.pad(image, pad_vector, mode = mode)
    
    return image
        
def select_roi_video(video_src, release = False):
    """Select a ROI from arbitrary video frame.
    
    Prompts the user to pause video and interactivelly select ROI.
    
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
    print ("Press key `p` to pause the video and select ROI")
    # Capture video
    vid = cv2.VideoCapture(video_src)
    count = 0
    try:
        while vid.isOpened():
            # Read next frame
            retval, frame = vid.read()
            # Exit when video can't be read
            if not retval: print ('Cannot read video file'); sys.exit();
            # Pause on 'p'
            if(cv2.waitKey(10) == ord('p')):
                #Define an initial bounding box
                # bbox = (158, 26, 161, 163)
                # Uncomment the line below to select a different bounding box
                pts, roi = select_roi(frame, win_name = "SelectROI")
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
        
    return pts, roi, vid, frame_pos, frame

def go_to_frame(vid, frame_pos, video_source, return_frame = False):
    """Jump to frame poisition in video
    
    Parameters
    ----------
    vid : cv2.VideoCapture object
        Already opened video object. If empty, video from video_source is read. 
    video_source : str
        path to video file
    frame_pos : float
        number of frame corresponding to current position in video [default = []]
    retrun_frame : 
        whether to also return next frame after setting position [default=True]
        
    Returns
    ----------
    ret_tuple : tuple
        pair of (vid, frame) or (vid, []) if return_frame is False
    
    """
    if (not vid) and video_source:
        vid = cv2.VideoCapture(video_source)  
    if not (frame_pos):
        num_frames = vid.get(cv2.CAP_PROP_FRAME_COUNT)
        frame_pos = np.int( 3 * num_frames / 4)
    # Later may add a check if video is already opened
    vid.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
    
    if return_frame:
        retval, frame = vid.read()
        if not retval: print ('Cannot read video file'); sys.exit();
        ret_tuple = (vid, frame)
    else: 
        ret_tuple = (vid, [])
        
    return ret_tuple
    
def find_contours(roi, contour_number = 1):
    """Find contours in a roi.
    
    Converts ROI of an image to grayscale, applies blurring and finds the x-th 
    most salient contour in the image, where x can be defiend by user.
    
    Parameters
    -----------
    roi : 2D array
        image to find contours in
    contour_number : int
        Contour position in a list ordered by area-size in decreasing order.
    
    Returns
    ------------
    gray_sharp : array_like
        gray-scale converted roi
    max_cnt : list
        list of points defining selected contour
    """
    # convert to grayscale, and blur
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = apply_pad(gray, (300, 50), "symmetric")
    gray_sharp = gray.copy()
    # Sigma and ksize are scale dependent!!! 
    gray = cv2.GaussianBlur(gray, ksize = (3, 3), sigmaX = 0)
    #gray = cv2.bilateralFilter(gray, d = 11, sigmaColor = 17, sigmaSpace = 17)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    edged = cv2.Canny(gray, threshold1 = 0, threshold2 = 255)
    edged = cv2.dilate(edged, kernel = kernel, iterations=1)
    # edged = cv2.erode(edged, kernel = None, iterations=1)
    
    cnts = cv2.findContours(
                    edged.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[1]
    # sort by area, keep top 10
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10] 

    max_cnt = cnts[contour_number]
    # Keep only clean part of circle. Try to automate later.
    # if you dont pad, use 170:-170
    max_cnt = max_cnt[400:-400]
    
    return max_cnt, gray_sharp

def fit_ellipse(contour, image):
    """Fits ellipse to contour.
    
    Fits ellipse to contour and draws it on image.
    
    Parameters
    ---------
    contour : list
        list of points defining selected contour
    image : 2D array
        Grayscale image for visualization
    
    Returns
    --------
    ellip : tuple
        Ellipse defined by centroid, axes lengths and angle.
    contourVis : ndarray
        image with contour drawn on it
    ellipVis : ndarray
        image with ellip drawn on it
    """
    ellip = cv2.fitEllipse(contour)
    # Visualize contour
    contourVis = image.copy() # preserve working image
    cv2.drawContours(contourVis, [contour], contourIdx = -1, 
                    color = (255, 255, 255), thickness =  3)
    # Visualize ellipse
    ellipVis = image.copy() # preserve working image
    cv2.ellipse(ellipVis, ellip, color = (255, 255, 255), thickness = 7)
    cv2.circle( ellipVis, (np.int(ellip[0][0]), np.int(ellip[0][1])),
                radius = 3, color = (255, 0, 0), thickness = 5)

    return ellip, contourVis, ellipVis
    
def mask_box_ellip(image, ellip):
    """Creates boolean mask of the ellipse and finds minimal-area rectangle.
    
    Parameters
    ------------
    image : 2D array
        region of interest as grayscale image
    ellip : tuple
        Ellipse defined by its centroid, axes lengths and angle.
    
    Returns
    ------------
    box : array_like
        corners of the minimal-area rectangle in 
        [top_left, top_right, bottom_right, bottom_left] order
    mask : 2D array
        boolean mask of the full ellipse contour
    imageVis : 2D array
        image with the minimal-area rectangle drawn on it
    """
    imageVis = image.copy()
    mask = np.zeros_like(image, dtype = np.uint8)
    cv2.ellipse(mask, ellip, color = (255, 255, 255), thickness = 7)
    # [0] selects the longest contour
    contour = cv2.findContours( mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)[1][0]
    
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int64(box)
    # obtain a consistent order of the points
    box = order_points(box)
    
    cv2.polylines(  img = imageVis, pts = [box], isClosed = True,
                    color = (255, 255, 255), thickness = 7)
    cv2.ellipse(imageVis, ellip, color = (255, 255, 255), thickness = 7)
    
    return box, mask, imageVis
    
def max_width_height(box):
    """Computes maximal width and height of rotated and possibly skewed
    bounding box
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

def four_point_transform(image, box, mask):
    """Obtain bird's eye view of an image.
    
    Obtains bird's eye view of an image by first projective-transforming it to
    fully span the extents of the frame (also includes rotation) and then
    applying uniaxial stretch to rectify an ellipse in the image into a circle.

    Parameters
    ---------
    image : 2D array
        grayscale image of region of interest
    box : array_like
        corners of the minimal-area rectangle in 
        [top_left, top_right, bottom_right, bottom_left] order
    mask : 2D array
        boolean mask of the full ellipse contour
    
    Returns
    ---------
    warped : 2D array
        projective and affine transformed image
    
    References
    ------------
   http://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
    """
    (maxWidth, maxHeight) = max_width_height(box)

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([[0, 0],
                    [maxWidth - 1, 0],
                    [maxWidth - 1, maxHeight - 1],
                    [0, maxHeight - 1]], dtype = "float32")

    # rows = maxHeight
    # cols = maxWidth
    # Get shape of initial image
    rows, cols = mask.shape
    # compute the perspective transform matrix and then apply it
    
    M = cv2.getPerspectiveTransform(box.astype(dtype = np.float32), dst)
    warped_temp = cv2.warpPerspective(image, M, (cols, rows))
    warped_mask_temp = cv2.warpPerspective(mask, M, (cols, rows))
    
    M_affine = affine_transform_warped(warped_mask_temp)
    
    # rows = np.int(rows*M_affine[1,1])
    # cols = np.int(cols*M_affine[0,0])
    # Get shape of initial image
    rows, cols = warped_mask_temp.shape
    
    # Apply transform and enlarge the output image as expected
    warped =        cv2.warpAffine(warped_temp, M_affine, (cols, rows))                      
    warped_mask =   cv2.warpAffine(warped_mask_temp, M_affine, (cols, rows))    
    return (warped, warped_mask, warped_temp, warped_mask_temp)
  
def affine_transform_warped(warped_mask):
    """Find affine transformation matrix.
    
    Finds the longest contour in binary mask and fits a circle and an ellipse
    to it. The circle and the ellipse are the compared and and affine mapping
    between the two is found. Used together with getcoords.four_point_transform.
    
    Parameters
    ---------
    warped_mask : 2D array
        a boolean mask after projective transform
    
    Returns
    ----------
    M : 2x2 array
        affine transformation matrix
    """
    contour = cv2.findContours( warped_mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)[1][0]
    
    circ = cv2.minEnclosingCircle(contour)
    (c, r) = circ
    
    ellip = cv2.fitEllipse(contour)
    majAx = ellip[1][1]
    minAx = ellip[1][0]
    angle_rad = (90 - ellip[2]) * (2*np.pi / 360)
    # Get directional shifts
    xshift = np.sin(angle_rad) * (majAx / 2)
    yshift = -np.sin(angle_rad) * (minAx / 2)

    # Circle points
    pts1 = np.float32([ [c[0] + r, c[1]],
                    [c[0] - r, c[1]],
                    [c[0], c[1] + r],
                    [c[0], c[1] - r]])

    # Ellipse points
    pts2 = np.float32([
                        [ellip[0][0] + majAx / 2, ellip[0][1] + yshift],
                        [ellip[0][0] - majAx / 2, ellip[0][1] - yshift],
                        [ellip[0][0] + xshift, ellip[0][1] + minAx / 2],
                        [ellip[0][0] - xshift, ellip[0][1] - minAx / 2]])
    M = cv2.getAffineTransform(pts2[0:3], pts1[0:3])
    rows, cols = warped_mask.shape
    M = np.round(M, decimals = 3)
    return M
    
def projective_transform(image, mask, D):
    """Obtain and apply projective transformtion
    
    Requires roi to have sufficient padding!
    
    Parameters
    ---------
    image : 2D array
        grayscale image of region of interest
    mask : 2D array
        boolean mask of the full ellipse contour
    D : float
        diameter of circular object in real-world dimensions [mm]
    
    Returns
    -----------
    M : 3x3 array
        projective transformation matrix
    warped : 2D array
        projective transformed image
    """
    contour = cv2.findContours( mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)[1][0]
    
    circ = cv2.minEnclosingCircle(contour)
    (c, r) = circ
    
    ellip = cv2.fitEllipse(contour)
    majAx = ellip[1][1]
    minAx = ellip[1][0]
    angle_rad = (90 - ellip[2]) * (2*np.pi / 360)
    # Get directional shifts
    xshift = np.sin(angle_rad) * (majAx / 2)
    yshift = -np.sin(angle_rad) * (minAx / 2)

    # Circle points
    pts_circ = np.float32([ [c[0] + r, c[1]],
                    [c[0] - r, c[1]],
                    [c[0], c[1] + r],
                    [c[0], c[1] - r]])

    # Ellipse points
    pts_ellip = np.float32([
                        [ellip[0][0] + majAx / 2, ellip[0][1] + yshift],
                        [ellip[0][0] - majAx / 2, ellip[0][1] - yshift],
                        [ellip[0][0] + xshift, ellip[0][1] + minAx / 2],
                        [ellip[0][0] - xshift, ellip[0][1] - minAx / 2]])
    M = cv2.getPerspectiveTransform(pts_ellip, pts_circ)    
    # debug_points(mask, ellip, circ, pts1, pts2)
    rows, cols = mask.shape
    warped_mask = cv2.warpPerspective(mask, M, (cols, rows))
    warped = cv2.warpPerspective(image, M, (cols, rows))
    
    scaling_factor = find_scale(warped_mask, D = D)
    np.save("projection_matrix.npy", M)
    np.save("scaling_factor.npy", scaling_factor)
        
    return M, scaling_factor, warped, warped_mask
    
def find_scale(warped_mask, D):
    """ Find scaling between image and real world dimensions
    
    Parameters
    -------------
    warped_mask : ndarray
        rectified image of a circular object
    D : float
        diameter of circular object in real-world dimensions [mm]
        
    Returns
    -------------
    scaling : float
        scaling factor [mm / pixel]
    """
    contour = cv2.findContours( warped_mask.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)[1][0]
    perim = cv2.arcLength(contour, closed = True)
    diameter = perim / np.pi
    scaling = D / diameter
    return scaling
    
    
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
    
def main():
    video_src = "videos/test.avi"
    pts, roi, vid, frame_pos, _ = select_roi_video(video_src)
    contour, gray = find_contours(roi)
    ellip, _, _= fit_ellipse(contour, gray)
    box, mask, _ = mask_box_ellip(gray, ellip)
    M, sf, _, _ = projective_transform(gray, mask, D = 1000)
    
if __name__ == '__main__':
    main()