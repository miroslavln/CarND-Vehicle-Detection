import numpy as np
import cv2
from skimage.feature import hog

def get_hog_features(image, orientations=9, pixels_per_cell=8, cells_per_block=2, viz=False):
    '''
    :param image: the image for the hog feature extraction
    :param orientations: number of orientation bins
    :param pixels_per_cell: size of a pixel
    :param cells_per_block: number of cells in each block
    :param viz: True if used for visualization, False if only a feature vector is needed
    :return: return the hog feature vector
    '''
    if viz:
        return hog(image[:, :, 0], orientations, pixels_per_cell=(pixels_per_cell, pixels_per_cell),
                   cells_per_block=(cells_per_block, cells_per_block),
                   visualise=True, feature_vector=False)

    return hog(image[:, :, 0], orientations, pixels_per_cell=(pixels_per_cell, pixels_per_cell),
               cells_per_block=(cells_per_block, cells_per_block),
               visualise=False, feature_vector=True)


def convert_color(img, cspace='LUV'):
    '''
    Converts the image to a given color space
    :param img: the image to be converted
    :param cspace: a string representing a color space [RGB,LUV,HLS,YUV] the image is assumed to be in RGB
    :return: the converted image
    '''
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
    '''
    Converts an image into a color histogram
    :param img: the image to be converted
    :param bins: number of bins to be used for the histogram
    :param bins_range: the color range to be used
    :return: returns an concatenation of the bins from each channel
    '''
    channels = []
    h, w, c = img.shape
    for i in range(c):
        channel = np.histogram(img[:, :, i], bins=bins, range=bins_range)
        channels.append(channel[0])
    return np.concatenate(channels)


def bin_spatial(img, size=(32, 32)):
    return cv2.resize(img, size).ravel()


def get_features(img):
    '''
    Creates a feature fector based on hog, stapital and color histogram
    :param img: the image to be converted to a feature vector
    :return: a feature vector containing the extracted features
    '''
    feature_image = convert_color(img)

    hog_features = get_hog_features(feature_image)
    spatial_features = bin_spatial(feature_image)
    color_features = color_histogram(feature_image)
    return np.concatenate((hog_features, spatial_features, color_features))


def load_image(img_path):
    '''
    Loads an image to be from a given path into a RGB color space
    :param img_path:
    :return: a numpy array containing the image
    '''
    img = cv2.imread(img_path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def extract_features(image_paths):
    '''
    Extracts the features from images
    :param image_paths: an array containing the images
    :return: the feature vector one for each image
    '''
    features = []
    for image_path in image_paths:
        img = load_image(image_path)
        features.append(get_features(img))

    return features