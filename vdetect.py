'''
Core functions for vehicle detection
'''

import os
from glob import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def open_image(fname, convert_to_rgb=True):

    im = cv2.imread(fname)

    if len(im.shape) == 2:
        return im

    if not convert_to_rgb:
        return im

    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def create_image_files_list(data_dir):
    '''
    Visit subdirectories of data_dir
    and build-up a list of file paths to images
    '''

    imfiles = []
    for el in os.listdir(data_dir):
        sub = os.path.join(data_dir, el)
        if os.path.isdir(sub):
            imfiles += glob(os.path.join(sub, '*.png'))

    return imfiles



def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    '''
    bboxes -- NumPy array (n x 4), where each row represents
              a rectangular region as two oposing points
              (x1, y1, x2, y2)
    '''

    res = np.copy(img)

    for x1, y1, x2, y2 in bboxes:
        cv2.rectangle(res, (x1, y1), (x2, y2), color, thick)

    return res


def window_region(im, bbox):

    x1, y1, x2, y2  = bbox

    return im[y1:y2+1, x1:x2+1]

def set_window_to_val(im, bbox, val):

    x1, y1, x2, y2  = bbox
    im[y1:y2+1, x1:x2+1] = val


def window_loop(side_len, step, x0, y0, x1, y1):
    '''
    Creates a NumPy array of square regions (with side side_len),
    such that the sequnce of regions (rows) in the array corresponds
    to a row-by-row sliding window loop, with the specified step,
    over the image domain defined by (x0, y0, x1, y1).
    '''

    def num_alternating_windows(total_len):
        n1 = total_len // side_len
        n2 = (total_len - step) // side_len
        return n1, n2

    def seq(val0, val1, n1, n2):

        assert (n1 == n2) or (n1 - n2 == 1)

        res = []

        s1 = range(
            val0,
            val0 + side_len * n1,
            side_len
        )

        s2 = range(
            val0 + step,
            val0 + step + side_len * n2,
            side_len
        )

        for a, b in zip(s1, s2):
            res.append(a)
            res.append(b)

        if n1 > n2:
            res.append(s1[-1])

        return res

    w = x1 - x0
    h = y1 - y0

    xn1, xn2 = num_alternating_windows(w)
    yn1, yn2 = num_alternating_windows(h)

    xseq = seq(x0, x1, xn1, xn2)
    yseq = seq(y0, y1, yn1, yn2)

    bboxes = []
    for y in yseq:
        for x in xseq:
            opposite_x = x + side_len - 1
            opposite_y = y + side_len - 1
            bboxes.append( [x, y, opposite_x, opposite_y] )

    return np.array(bboxes)


def define_main_region_custom():
    '''
    Return the custom main region of interest
    for the project's images and video.
    The region "cuts" sky up and the car's trunk
    down, and has the dimensions of (1280, 512)
    '''

    x_max = 1280

    y_bottom = 650
    y_top = y_bottom - 512

    return 0, y_top, x_max, y_bottom


def sliding_window(im, func):

    rows, cols = im.shape[:2]
    canvas = np.zeros((rows, cols))

    main_region = define_main_region_custom()
    mr_x0, mr_y0, mr_x1, mr_y1 = main_region

    loops = [
        window_loop(512, 256, mr_x0, mr_y0, mr_x1, mr_y1),
        window_loop(256, 128, mr_x0, mr_y0, mr_x1, mr_y1),
        window_loop(128, 64, mr_x0, mr_y0+128, mr_x1, mr_y1-64),
        window_loop(64, 32, mr_x0, mr_y0+128+64, mr_x1, mr_y1-64-64)
    ]

    for loop in loops:
        for bbox in loop:
            win = window_region(im, bbox)
            res = func(win)

            if res:
                set_window_to_val(canvas, bbox, 1)


def convert_colorspace_and_get_channels(im, conversion):

    converted = cv2.cvtColor(im, conversion)

    n_channels = converted.shape[2]

    channels = [converted[:,:,i] for i in range(n_channels)]
    return channels


def extract_hog_features(im, n_orient=9, cell_sz=8, block_sz=2):

    features = hog(
        im,
        orientations=n_orient,
        pixels_per_cell=(cell_sz, cell_sz),
        cells_per_block=(block_sz, block_sz),
        block_norm='L2-Hys',
        visualise=False
    )

    return features


def spatial_binning(im, size=32):

    resized_im = cv2.resize(im, (size, size))
    return resized_im.ravel()


def image_histogram(im, n_bins=32):

    counts, _ = np.histogram(im, bins=n_bins, range=(0, 256))
    return counts


def extract_features(im):

    gray =  cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)

    LUV = convert_colorspace_and_get_channels(im, cv2.COLOR_RGB2LUV)
    HLS = convert_colorspace_and_get_channels(im, cv2.COLOR_RGB2HLS)

    H = HLS[0]
    U = LUV[1]
    R = im[:, :, 0]
    G = im[:, :, 1]
    B = im[:, :, 2]

    hog_features = extract_hog_features(gray)
    binned_pixels = spatial_binning(gray)

    histograms = []
    for channel in (R, G, B, H, U):
        hist = image_histogram(channel)
        histograms.append(hist)

    hist_vec = np.hstack(histograms)

    return hog_features, binned_pixels, hist_vec


def scale_data(*data_subsets):
    '''
    Scale data subsets using StandardScaler that is fit using the
    first subset (typically, the training subset)
    '''

    first = data_subsets[0]
    scaler = StandardScaler().fit(first)

    scaled_subsets = [scaler.transform(x) for x in data_subsets]
    return scaled_subsets


def split_and_scale_features(list_of_feature_vecs):

    X_all = np.array(list_of_feature_vecs, dtype=np.float64)
    X_train, X_test = train_test_split(X_all)

    X_train_scaled, X_test_scaled = scale_data(X_train, X_test)
    return X_train_scaled, X_test_scaled


def prepare_train_test_data_single_class(imfiles, yval):
    '''
    Prepare features from the supplied image files
    characterized with a fixed y value (0 or 1).
    Returns training and testing data sets
    (X_train, y_train, X_test, y_test)
    '''

    feature_sets = ('hog', 'binned', 'hist')
    lists_of_feature_vecs = {k: [] for k in feature_sets}

    for imfile in imfiles:

        im = open_image(imfile)

        hog, binned, hist = extract_features(im)

        lists_of_feature_vecs['hog'].append(hog)
        lists_of_feature_vecs['binned'].append(binned)
        lists_of_feature_vecs['hist'].append(hist)

    train = dict()
    test = dict()
    for k, lst in lists_of_feature_vecs.items():
        x_train, x_test = split_and_scale_features(lst)
        train[k] = x_train
        test[k] = x_test

    X_train = np.hstack([train[k] for k in feature_sets])
    X_test = np.hstack([test[k] for k in feature_sets])

    y_train = np.ones(X_train.shape[0]) * yval
    y_test = np.ones(X_test.shape[0]) * yval

    return X_train, y_train, X_test, y_test


def prepare_train_test_data(imfiles_0, imfiles_1):
    '''
    Preprocess all image files (for both y=0 and y=1 cases)
    and return training and testing data sets
    (X_train, y_train, X_test, y_test)
    '''

    X0_train, y0_train, X0_test, y0_test = prepare_train_test_data_single_class(imfiles_1, 0)
    X1_train, y1_train, X1_test, y1_test = prepare_train_test_data_single_class(imfiles_1, 1)

    X_train = np.vstack((X0_train, X1_train))
    X_test = np.vstack((X0_test, X1_test))

    y_train = np.hstack((y0_train, y1_train))
    y_test = np.hstack((y0_test, y1_test))

    return X_train, y_train, X_test, y_test
