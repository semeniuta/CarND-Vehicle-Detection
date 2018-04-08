'''
Core functions for vehicle detection
'''

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


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

        s1 = range(val0, side_len * n1, side_len)
        s2 = range(val0 + step, step + side_len * n2, side_len)

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


def sliding_window():
    pass
