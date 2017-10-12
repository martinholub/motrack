import numpy as np
#     s_size = hsv_upperb[1] - hsv_lowerb[1]
#     v_size = hsv_upperb[2] - hsv_lowerb[2]
#     s_range = np.arange(hsv_lowerb[1], hsv_upperb[1] + 1).tolist()
#     v_range = np.arange(hsv_lowerb[2], hsv_upperb[2] + 1).tolist()
#     # h_ranges = [[s_range], [v_range]]
#     h_sizes =  [s_size, v_size]
#     h_ranges = [hsv_lowerb[1], hsv_upperb[1], hsv_lowerb[2], hsv_upperb[2]]

# Draw bbox on the average image
# p1 = (int(bbox[0]), int(bbox[1]))
# p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
# frame_avg = cv2.rectangle(frame_avg, p1, p2, color = red)

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

def debug_plot2(frame, pts, roi = np.empty(0)):
    """Helper debugging plot
    """
    import matplotlib.pyplot as plt
    if not roi.size:
        roi = frame[pts[0][0]:pts[1][0], pts[0][1]:pts[2][1],:]
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
    
# def pts_to_bbox(pts):
    # """Converts 4 x,y coordinate pairs to corresponding bbox tuple
    # """
    # pts = getcoords.order_points(pts)
    # bbox = (pts[0], pts[1], pts[2] - pts[0], pts[3] - pts[1])
    # return bbox
    
    
    
        # elif type == "threshold":
        # try:
            # (frame_vis, props) =    label_contours(prob_mask, 
                                    # kwargs["kSize_gauss"], kwargs["sigmaX"],
                                    # kwargs["kSize_canny"])
        # except IndexError:
            # return None
        
        # bbox = (props.bbox[1], props.bbox[0],
                # np.int(props.bbox[3]-props.bbox[1]), np.int(props.bbox[2] - props.bbox[0]))
        # pts = bbox_to_pts(bbox)
        # centroid = props.centroid
        
    # elif type == 3:
        # from skimage import measure
        # binary = np.where(frame > np.max(frame) * 0.25, 255, 0).astype(np.uint8)
        # binary = cv2.erode(binary, kernel = kernel, iterations = 1)
        # binary = cv2.dilate(binary, kernel = kernel, iterations = 5)
        
        # labels = measure.label(binary)
        # props = measure.regionprops(labels)
        # props = sorted(props, key = lambda x: x.area , reverse = False)[:1]
        
        # for prop in props:
            # try:
                # label_mask = np.where(labels == prop.label, 255, 0).astype(np.uint8)
                # vis_img = cv2.add(vis_img, label_mask)
            # except IndexError:
                # pass
        
        # labels = measure.label(vis_img)
        # cnts = measure.regionprops(labels)[0]
    # elif type == 4:
        # # indices = np.where(frame > np.max(frame) * 0.5)
        # # from sklearn.metrics.pairwise import pairwise_distances
        # # dist_mat = pairwise_distances(indices, metric = "euclidean")
        # # from sklearn.cluster import DBSCAN
        # # db = DBSCAN(eps = .3, min_samples = 2).fit(dist_mat)    
        # #
        # pass