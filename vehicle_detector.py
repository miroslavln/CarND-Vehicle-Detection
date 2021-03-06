import numpy as np
import cv2
from features import get_features
from scipy.ndimage.measurements import label

class VehicleDetector:
    def __init__(self, clf, scaler, overlap):
        self.heatmap = None
        self.ytop = 400
        self.ybottom = 650
        self.sizes = [(140, 650, 0.5),
                      (120, 600, 0.7),
                      (96, 500, 0.7),
                      (64, 450, 0.7)]

        self.clf = clf
        self.scaler = scaler
        self.overlap = overlap

    @staticmethod
    def draw_boxes(img, bboxes):
        '''
        Draws bounding boxes on the provided image
        :param img: the image to have the bounding boxes drawn
        :param bboxes: array containing the bounding boxes coordinates
        :return: an image with bounding boxes drawn
        '''
        res = np.copy(img)
        for top, bottom in bboxes:
            cv2.rectangle(res, top, bottom, color=(0, 0, 255), thickness=6)
        return res

    @staticmethod
    def get_windows(img, y_start_stop, window_size, overlap=0.5):
        '''
        Splits an image into multiple overlapping windows
        :param img: an image that describes the area of the bounding boxes
        :param y_start_stop: the start and end position on the horizontal
        :param window_size: the size of the window
        :param overlap: the overlap coefficient.
        :return: a list of coordinates for the overlapping windows
        '''
        h, w, c = img.shape
        xspan = w
        yspan = y_start_stop[1] - y_start_stop[0]

        step = int(window_size * (1.0 - overlap))
        x_windows = (xspan // step)
        y_windows = (yspan // step)

        windows = []
        for y in range(y_windows):
            for x in range(x_windows):
                top_x = x * step
                top_y = y_start_stop[0] + y * step
                bottom_x = top_x + window_size
                bottom_y = top_y + window_size
                windows.append(((top_x, top_y), (bottom_x, bottom_y)))

        return windows

    def search_windows(self, img, windows):
        '''
        Detects if a car is in any of the windows using the classifier
        :param img: the image to be checked
        :param windows: list of the windows that we want to search through
        :return: a list of window coordinates where a car was detected
        '''
        features = []
        for window in windows:
            x1, y1 = window[0]
            x2, y2 = window[1]
            test_img = img[y1:y2, x1:x2, :]
            assert test_img is not None
            test_img = cv2.resize(test_img, (64, 64))
            features.append(get_features(test_img))

        features_tensor = np.array(features)
        features_tensor = self.scaler.transform(features_tensor)
        prediction = self.clf.predict(features_tensor)

        return [windows[i] for i in range(len(windows)) if prediction[i] == 1]

    def add_heat(self, bboxes, heatmap):
        '''
        Adds heat to the heatmap based on the provided bounding boxes
        :param bboxes: boxes where to add head to the heatmap
        :param heatmap: the initial heatmap
        :return: returns the augmented heatmap
        '''
        for box in bboxes:
            x1, y1 = box[0]
            x2, y2 = box[1]
            heatmap[y1:y2, x1:x2] += 1
        return heatmap

    def apply_threashold(self, tresh):
        self.heatmap[(self.heatmap < tresh)] = 0

    @staticmethod
    def get_labeled_boxes(labels):
        '''
        Gets the square regions where cars were detected
        :param labels: a list of regions
        :return: list of windows covering the detected cars
        '''
        bboxes = []
        for car in range(labels[1]):
            nonzero = (labels[0] == car + 1).nonzero()

            nonzero_x = np.array(nonzero[1])
            nonzero_y = np.array(nonzero[0])

            top_x, top_y = np.min(nonzero_x), np.min(nonzero_y)
            bottom_x, bottom_y = np.max(nonzero_x), np.max(nonzero_y)

            bboxes.append(((top_x, top_y), (bottom_x, bottom_y)))
        return bboxes

    def process_image(self, img):
        '''
        Detect the cars ina an image and draw bounding boxes around them
        :param img: the image to be processed
        :return: a new image with bounding boxes drawn where cars are detected
        '''
        if self.heatmap is None:
            self.heatmap = np.zeros_like(img[:, :, 0])

        heatmap = np.zeros_like(img[:, :, 0])

        detected_img = np.copy(img)

        for i, size in enumerate(self.sizes):
            windows_size, ybottom, overlap = size
            y_start_stop = (self.ytop, ybottom)

            windows = self.get_windows(img, y_start_stop, windows_size, overlap)
            detected = self.search_windows(img, windows)
            heatmap = self.add_heat(detected, heatmap)
            detected_img = self.draw_boxes(detected_img, detected)

        self.heatmap = self.heatmap / 2.0
        self.heatmap += heatmap
        self.apply_threashold(4)
        labels = label(self.heatmap)
        bboxes = self.get_labeled_boxes(labels)

        res = self.draw_boxes(img, bboxes)
        res = self.display_detections(res, detected_img)

        return res

    def display_detections(self, img, mask):
        '''
        Draws a scaled version of the image in the upper right corner
        :param img: the image to overlay the smaller one over
        :param mask: the upper right image
        :return: a new image with the smaller image in the upper right corner
        '''
        mask = cv2.resize(mask, None, fx=1 / 4, fy=1 / 4, interpolation=cv2.INTER_CUBIC)
        return self.add_display(img, mask, x_offset=img.shape[1]*0.75, y_offset=10)

    def add_display(self, result, display, x_offset, y_offset):
        '''
        Puts an image in the provided x and y offset
        :param result: the image to have the display overlayed over
        :param display: the image do overlay on top
        :param x_offset: the start offset in the horizontal
        :param y_offset: the start offset in the vertical
        :return: the generated image with the overlay drawn
        '''
        result[y_offset: y_offset + display.shape[0], x_offset: x_offset + display.shape[1]] = display
        return result
