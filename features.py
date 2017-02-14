import numpy as np
import cv2
from skimage.feature import hog

def get_hog_features(image, orientations=9, pixels_per_cell=8, cells_per_block=2, viz=False):
    if viz:
        return hog(image[:, :, 0], orientations, pixels_per_cell=(pixels_per_cell, pixels_per_cell),
                   cells_per_block=(cells_per_block, cells_per_block),
                   visualise=True, feature_vector=False)

    return hog(image[:, :, 0], orientations, pixels_per_cell=(pixels_per_cell, pixels_per_cell),
               cells_per_block=(cells_per_block, cells_per_block),
               visualise=False, feature_vector=True)


def convert_color(img, cspace='LUV'):
    if cspace == 'RGB':
        return img
    if cspace == 'HSV':
        feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    elif cspace == 'LUV':
        feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    elif cspace == 'HLS':
        feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    elif cspace == 'YUV':
        feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    return feature_image


def color_histogram(img, bins=32, bins_range=(0, 256)):
    channels = []
    h, w, c = img.shape
    for i in range(c):
        channel = np.histogram(img[:, :, i], bins=bins, range=bins_range)
        channels.append(channel[0])
    return np.concatenate(channels)


def bin_spatial(img, size=(32, 32)):
    return cv2.resize(img, size).ravel()


def get_features(img):
    feature_image = convert_color(img)

    hog_features = get_hog_features(feature_image)
    spatial_features = bin_spatial(feature_image)
    color_features = color_histogram(feature_image)
    return np.concatenate((hog_features, spatial_features, color_features))


def load_image(img_path):
    img = cv2.imread(img_path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def extract_features(image_paths):
    features = []
    for image_path in image_paths:
        img = load_image(image_path)
        features.append(get_features(img))

    return features