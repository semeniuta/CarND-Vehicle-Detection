'''
Core functions for vehicle detection
'''

import os
from glob import glob
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import json

from scaler import FeatureScaler


FEATURE_SETS = ('hog', 'binned', 'hist')


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


def increment_window(im, bbox, increment_val=1):

    x1, y1, x2, y2  = bbox

    w = abs(x2 - x1 + 1)
    h = abs(y2 - y1 + 1)

    increment_arr = np.ones((h, w)) * increment_val

    im[y1:y2+1, x1:x2+1] += increment_arr


def window_loop(side_len, step, x0, y0, x1, y1):

    bboxes = []

    y = y0
    while y < y1:
        y_other = y + (side_len - 1)

        if y_other > y1:
            break

        x = x0
        while x < x1:

            x_other = x + (side_len - 1)

            if x_other > x1:
                break

            bboxes.append(
                np.array([x, y, x_other, y_other])
            )

            x += step

        y += step

    return np.array(bboxes)


def window_loop_old(side_len, step, x0, y0, x1, y1):
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


def define_loops_custom_1():

    main_region = define_main_region_custom()
    mr_x0, mr_y0, mr_x1, mr_y1 = main_region

    loops = [
        window_loop(512, 256, mr_x0, mr_y0, mr_x1, mr_y1),
        window_loop(256, 128, mr_x0, mr_y0, mr_x1, mr_y1),
        window_loop(128, 64, mr_x0, mr_y0+128, mr_x1, mr_y1-64),
        window_loop(64, 32, mr_x0, mr_y0+128+64, mr_x1, mr_y1-64-64)
    ]

    return loops

def define_loops_custom_2():

    main_region = define_main_region_custom()
    mr_x0, mr_y0, mr_x1, mr_y1 = main_region

    loops = [
        window_loop(256, 64, mr_x0, mr_y0, mr_x1, mr_y1),
        window_loop(128, 32, mr_x0, mr_y0+128, mr_x1, mr_y1-64),
        window_loop(64, 16, mr_x0, mr_y0+128+64, mr_x1, mr_y1-64-64)
    ]

    return loops


def define_main_region_custom_2():

    x_left = 6 * 64
    x_max = 1280

    y_bottom = 650
    y_top = y_bottom - (512 - 128 - 64)

    return x_left, y_top, x_max, y_bottom


def define_loops_custom_3():

    main_region = define_main_region_custom_2()
    mr_x0, mr_y0, mr_x1, mr_y1 = main_region

    loops = [
        window_loop(256, 64, mr_x0, mr_y0, mr_x1, mr_y1),
        window_loop(128, 32, mr_x0, mr_y0, mr_x1, mr_y1-64),
        window_loop(64, 16, mr_x0, mr_y0+64, mr_x1, mr_y1-128)
    ]

    return loops


def sliding_window(im, loops, extract, classifiers):

    rows, cols = im.shape[:2]
    canvas = np.zeros((rows, cols))

    for loop in loops:
        for bbox in loop:
            win = window_region(im, bbox)

            if win.shape != (64, 64, 3):
                win = cv2.resize(win, (64, 64))

            features = extract(win)

            for clf in classifiers:
                yhat = clf.predict(features)[0]

                if yhat == 1.:
                    increment_window(canvas, bbox)

    return canvas


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


def extract_features(
    im,
    hog_n_orient=9,
    hog_cell_sz=8,
    hog_block_sz=2,
    binning_sz=32,
    hist_bins=32,
):

    gray =  cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)

    LUV = convert_colorspace_and_get_channels(im, cv2.COLOR_RGB2LUV)
    HLS = convert_colorspace_and_get_channels(im, cv2.COLOR_RGB2HLS)

    H = HLS[0]
    U = LUV[1]
    R = im[:, :, 0]
    G = im[:, :, 1]
    B = im[:, :, 2]

    hog_features = extract_hog_features(gray, hog_n_orient, hog_cell_sz, hog_block_sz)
    binned_pixels = spatial_binning(gray, binning_sz)

    histograms = []
    for channel in (R, G, B, H, U):
        hist = image_histogram(channel, hist_bins)
        histograms.append(hist)

    hist_vec = np.hstack(histograms)

    features = {
        'hog': hog_features,
        'binned': binned_pixels,
        'hist': hist_vec
    }

    return features


def extract_features_2(
    im,
    hog_n_orient=9,
    hog_cell_sz=8,
    hog_block_sz=2,
    binning_sz=32,
    hist_bins=32,
):

    YCrCb = cv2.cvtColor(im, cv2.COLOR_BGR2YCrCb)
    Cr = YCrCb[:, :, 1]
    Cb = YCrCb[:, :, 2]

    LUV = convert_colorspace_and_get_channels(im, cv2.COLOR_RGB2LUV)
    HLS = convert_colorspace_and_get_channels(im, cv2.COLOR_RGB2HLS)

    H = HLS[0]
    U = LUV[1]
    R = im[:, :, 0]
    G = im[:, :, 1]
    B = im[:, :, 2]

    hog_features = extract_hog_features(Cb, hog_n_orient, hog_cell_sz, hog_block_sz)

    gray =  cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    binned_pixels = spatial_binning(gray, binning_sz)

    histograms = []
    for channel in (R, G, B, H, U, Cr, Cb):
        hist = image_histogram(channel, hist_bins)
        histograms.append(hist)

    hist_vec = np.hstack(histograms)

    features = {
        'hog': hog_features,
        'binned': binned_pixels,
        'hist': hist_vec
    }

    return features


def split_original_features(list_of_feature_vecs):
    '''
    Convert a list of feature vectors into
    a single NumPy array and split it
    into training and testing set
    '''

    X_all = np.array(list_of_feature_vecs, dtype=np.float64)
    X_train, X_test = train_test_split(X_all)

    return X_train, X_test


def prepare_train_test_data_single_class(
    imfiles,
    hyperparams,
    extract_features_func=extract_features
):
    '''
    Prepare features from the supplied image files
    characterized with a fixed y value (0 or 1).
    Returns training and testing dictionaries
    with 'hog', 'binned', 'hist' as keys and the correspoding
    (unscaled) data sets as values
    '''

    lists_of_feature_vecs = {k: [] for k in FEATURE_SETS}

    for imfile in imfiles:

        im = open_image(imfile)
        im_features = extract_features_func(im, **hyperparams)

        for k in FEATURE_SETS:
            lists_of_feature_vecs[k].append(im_features[k])

    train_orig = dict()
    test_orig = dict()
    for k, lst in lists_of_feature_vecs.items():
        train_orig[k], test_orig[k] = split_original_features(lst)

    return train_orig, test_orig


def prepare_train_test_data(
    imfiles_0,
    imfiles_1,
    hyperparams,
    extract_features_func=extract_features
):
    '''
    Preprocess all image files (for both y=0 and y=1 cases)
    and return training and testing data sets, and the FeatureScaler object
    (X_train, y_train, X_test, y_test, scaler)
    '''

    print('Extracting original heterogeneous features from images (class 0)')
    train_orig_0, test_orig_0 = prepare_train_test_data_single_class(
        imfiles_0,
        hyperparams,
        extract_features_func
    )

    print('Extracting original heterogeneous features from images (class 1)')
    train_orig_1, test_orig_1 = prepare_train_test_data_single_class(
        imfiles_1,
        hyperparams,
        extract_features_func
    )

    print('Scaling features')

    keys = train_orig_0.keys()
    train_orig_all = {k: np.vstack((train_orig_0[k], train_orig_1[k])) for k in keys}
    test_orig_all = {k: np.vstack((test_orig_0[k], test_orig_1[k])) for k in keys}

    scaler = FeatureScaler(train_orig_all)

    train_scaled = scaler.scale(train_orig_all)
    test_scaled = scaler.scale(test_orig_all)

    print('Final preparation of training and testing data sets')

    X_train = np.hstack(
        [train_scaled[k] for k in FEATURE_SETS]
    )

    X_test = np.hstack(
        [test_scaled[k] for k in FEATURE_SETS]
    )

    y_train = np.hstack((
        np.zeros( train_orig_0['hog'].shape[0] ),
        np.ones( train_orig_1['hog'].shape[0] ),
    ))

    y_test = np.hstack((
        np.zeros( test_orig_0['hog'].shape[0] ),
        np.ones( test_orig_1['hog'].shape[0] ),
    ))

    feature_sets_sizes = {k: train_orig_all[k].shape[1] for k in FEATURE_SETS}

    return X_train, y_train, X_test, y_test, scaler, feature_sets_sizes


def create_feature_extractor(scaler, hyperparams, extract_features_func=extract_features):

    def extract(im):
        features_dict = extract_features_func(im, **hyperparams)
        scaled_dict = scaler.scale(features_dict)

        x = np.hstack(
            [scaled_dict[k] for k in FEATURE_SETS]
        )

        return x

    return extract


def find_ccomp(im, *args, **kwargs):

    num, labels, stats, centroids = cv2.connectedComponentsWithStats(im, *args, **kwargs)

    stats_df = pd.DataFrame(stats, columns=['left', 'top', 'width', 'height', 'area'])
    stats_df['x'] = centroids[:,0]
    stats_df['y'] = centroids[:,1]

    return labels, stats_df


def mask_threashold_range(im, thresh_min, thresh_max):
    '''
    Return a binary mask image where pixel intensities
    of the original image lie within [thresh_min, thresh_max)
    '''

    binary_output = (im >= thresh_min) & (im < thresh_max)
    return np.uint8(binary_output)


def segment_vehicles(heatmap, threshold_ratio=0.7, low_limit=10):

    heatmap_max = np.max(heatmap)

    if heatmap_max <= low_limit:
        return None

    threshold = threshold_ratio * heatmap_max
    thresholded = np.array(heatmap >= threshold, dtype=np.uint8)

    labels, stats_df = find_ccomp(thresholded)

    n = len(stats_df) - 1
    bboxes = np.zeros((n, 4), dtype=np.int)

    df = stats_df.iloc[1:]
    for i in range(n):

        arr = np.array([
            df.iloc[i]['left'],
            df.iloc[i]['top'],
            df.iloc[i]['left'] + df.iloc[i]['width'],
            df.iloc[i]['top'] + df.iloc[i]['height']
        ])

        bboxes[i, :] = arr

    return bboxes


def load_ml_results(dir_ml, extract_features_func=extract_features):

    classifiers_file = os.path.join(dir_ml, 'classifiers.p')
    scaler_file = os.path.join(dir_ml, 'scaler.p')
    hp_file = os.path.join(dir_ml, 'hyper.json')

    hyperparams = load_json(hp_file)
    scaler = load_pickle(scaler_file)
    classifiers = load_pickle(classifiers_file)

    extract = create_feature_extractor(scaler, hyperparams, extract_features_func)

    return classifiers, extract, scaler, hyperparams


def select_classifiers(classifiers, clf_names=None):

    if clf_names is None:
        return classifiers.values()
    else:
        return [classifiers[name] for name in clf_names]


def load_pickle(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)


def load_json(fname):
    with open(fname, 'r') as f:
        return json.load(f)
