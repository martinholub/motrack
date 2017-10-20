alpha = 0.5 # Weighting for running sums
step = 5 # Number of steps accumulated in running sum

its = 10
eps = 1

dist_lim = 0
fraction = .25
connectivity = 8
min_area = 1

sigmaX = 1 # std of Gaussian kernel
kSize_gauss = (3, 3) # Gaussian kernel size
kSize_canny = (5, 5)

reinitialize_roi = True
reinitialize_hsv = True
reinitialize_bg = True
remove_bg = True
double_substract_bg = True
ext = ".dat"
chs = [1, 2]
h_sizes = [256, 256]
h_ranges = [0, 256, 0, 256]
check_area = False

height_resize = 500
annotate_mask = True
plot_mask = True
show_frames = True