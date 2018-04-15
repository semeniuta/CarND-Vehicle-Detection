import os
import matplotlib.image as mpimg
from matplotlib import pyplot as plt
import numpy as np
import cv2
from glob import glob
from moviepy.editor import VideoFileClip

import vdetect


def visualize_main_region(im_src, dir_output):

    im = mpimg.imread(im_src)

    main_region = vdetect.define_main_region_custom()

    viz = vdetect.draw_boxes(im, [main_region])
    mpimg.imsave(os.path.join(dir_output, 'mainROI.jpg'), viz)


def visualize_window_search(im_src, dir_output):

    im = mpimg.imread(im_src)

    main_region = vdetect.define_main_region_custom()
    mr_x0, mr_y0, mr_x1, mr_y1 = main_region

    plt.figure(figsize=(20, 12))

    plt.subplot(2, 2, 1)
    plt.imshow(
        vdetect.draw_boxes(
            im,
            vdetect.window_loop(512, 256, mr_x0, mr_y0, mr_x1, mr_y1)
        )
    )
    plt.title('size={:d}, step={:d}, main ROI'.format(512, 256))
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(
        vdetect.draw_boxes(
            im,
            vdetect.window_loop(256, 128, mr_x0, mr_y0, mr_x1, mr_y1)
        )
    )
    plt.title('size={:d}, step={:d},  main ROI'.format(256, 128))
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(
        vdetect.draw_boxes(
            im,
            vdetect.window_loop(128, 64, mr_x0, mr_y0+128, mr_x1, mr_y1-64)
        )
    )
    plt.title('size={:d}, step={:d}, smaller ROI'.format(128, 64))
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(
        vdetect.draw_boxes(
            im,
            vdetect.window_loop(64, 32, mr_x0, mr_y0+128+64, mr_x1, mr_y1-64-64)
        )
    )
    plt.title('size={:d}, step={:d}, even smaller ROI'.format(64, 32))
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(dir_output, 'winsearch.jpg'))


def visualize_heatmap(dir_images, dir_ml, dir_output):

    classifiers, extract, scaler, hyperparams = vdetect.load_ml_results(dir_ml)
    images = [mpimg.imread(f) for f in glob(dir_images + '/*.jpg')]

    plt.figure(figsize=(10, 20))
    idx = 1
    for im in images:

        swres = vdetect.sliding_window(im, extract, classifiers.values())

        plt.subplot(6, 2, idx)
        plt.imshow(im)
        plt.axis('off')
        idx += 1

        plt.subplot(6, 2, idx)
        plt.imshow(swres)
        plt.axis('off')
        plt.title('max={}'.format(np.max(swres)))
        idx += 1

    plt.tight_layout()
    plt.savefig(os.path.join(dir_output, 'heatmap.jpg'))


def visualize_classifiers(dir_images, dir_ml, dir_output):

    classifiers, extract, scaler, hyperparams = vdetect.load_ml_results(dir_ml)
    images = [mpimg.imread(f) for f in glob(dir_images + '/*.jpg')]

    n_cols = len(classifiers) + 1
    n_images = len(images)

    plt.figure(figsize=(20, 15))
    idx = 1
    for im in images:

        plt.subplot(n_images, n_cols, idx)
        plt.imshow(im)
        plt.axis('off')
        idx += 1

        for k, clf in classifiers.items():

            heatmap = vdetect.sliding_window(im, extract, [classifiers[k]])

            plt.subplot(n_images, n_cols, idx)
            plt.imshow(heatmap)
            plt.axis('off')
            plt.title(k)
            idx += 1

    plt.tight_layout()
    plt.savefig(os.path.join(dir_output, 'classifiers.jpg'))


if __name__ == '__main__':

    DIR_OUT = 'output_images'

    visualize_main_region('test_images/test3.jpg', DIR_OUT)
    visualize_window_search('test_images/test3.jpg', DIR_OUT)

    visualize_heatmap(
        'test_images',
        'serialize/2018-04-15_113152',
        DIR_OUT
    )

    visualize_classifiers(
        'test_images',
        'serialize/2018-04-15_113152',
        DIR_OUT
    )
