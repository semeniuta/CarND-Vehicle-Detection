# **Vehicle Detection**

## Writeup

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.


[//]: # (Image References)

[image1]: ./examples/car_not_car.png
---

### Code overview

The core functions used in this project are combined into `vdetect.py` Python module. `scaler.py` contains the `FeatureScaler` class used to perform scaling on different sets of features extracted from an image. `train.py` is a script that performs training of a number of classifiers on the vehicles/non-hevicles data. All the resulting media files are generated by the `genmedia.py` script.

### Feature engineering

This project uses a feature preparation method (see `vdetect.extract_features`) that combines three sets of features, extracted from a (64, 64, 3) image:

* HOG features, extracted from the grayscale transformation of the original image
* Binned version of the the grayscale transformation of the original image: (64, 64) -> (32, 32) -> flatten
* Combined histogram of color channels: R, G, B, H (of HLS), U (of LUV)

Upon extraction, the three feature sets are returned separately as values in a dictionary. Further, in `vdetect.prepare_train_test_data_single_class` and `vdetect.prepare_train_test_data`, dictionaries from individual images are combined together, shuffled, and separated into training and testing sets. To perform scaling on each feature sets separately, the `scaler.FeatureScaler` class is used. When constructed, `FeatureScaler` accepts a dictionary of training sets and fits a `sklearn.preprocessing.StandardScaler` on each feature set.

### Sliding window


### Machine learning


### Vehicles detection