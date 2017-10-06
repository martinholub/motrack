alpha = 0.5 # Weighting for running sums
step = 5 # Number of steps accumulated in running sum

its = 10
eps = 1

dist_lim = 20
fraction = .25
connectivity = 8
min_area = 3000

sigmaX = 1 # std of Gaussian kernel
kSize_gauss = (3, 3) # Gaussian kernel size
kSize_canny = (5, 5)
y_pad = 0 # optional padding of a roi

reinitialize_roi = False
reinitialize_hsv = False
reinitialize_bg = False
remove_bg = True
ext = ".dat"
save_init = False
load_params = not save_init
chs = [1, 2]
h_sizes = [256, 256]
h_ranges = [0, 256, 0, 256]

height_resize = 500
annotate_mask = False
plot_mask = False
show_frames = False