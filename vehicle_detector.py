import glob
import time

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
                      (120, 600, 0.5),
                      (96, 550, 0.7),
                      (64, 500, 0.7)]

        self.clf = clf
        self.scaler = scaler
        self.overlap = overlap

    @staticmethod
    def draw_boxes(img, bboxes):
        res = np.copy(img)
        for top, bottom in bboxes:
            cv2.rectangle(res, top, bottom, color=(0, 0, 255), thickness=6)
        return res

    @staticmethod
    def get_windows(img, y_start_stop, window_size, overlap=0.5):
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
        for box in bboxes:
            x1, y1 = box[0]
            x2, y2 = box[1]
            heatmap[y1:y2, x1:x2] += 1

    def apply_threashold(self, tresh):
        self.heatmap[(self.heatmap < tresh)] = 0

    @staticmethod
    def get_labeled_boxes(labels):
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
        if self.heatmap is None:
            self.heatmap = np.zeros_like(img[:, :, 0])

        heatmap = np.zeros_like(img[:, :, 0])

        detected_img = np.copy(img)

        for i, size in enumerate(self.sizes):
            windows_size, ybottom, overlap = size
            y_start_stop = (self.ytop, ybottom)

            windows = self.get_windows(img, y_start_stop, windows_size, overlap)
            detected = self.search_windows(img, windows)
            self.add_heat(detected, heatmap)
            detected_img = self.draw_boxes(detected_img, detected)

        self.heatmap = self.heatmap / 2.0
        self.heatmap += heatmap
        self.apply_threashold(4)
        labels = label(self.heatmap)
        bboxes = self.get_labeled_boxes(labels)

        res = self.draw_boxes(img, bboxes)
        res = self.display_heat(res, detected_img)

        return res

    def display_heat(self, img,  mask):
        mask = cv2.resize(mask, None, fx=1 / 4, fy=1 / 4, interpolation=cv2.INTER_CUBIC)
        return self.add_display(img, mask, x_offset=img.shape[1]*0.75, y_offset=10)

    def add_display(self, result, display, x_offset, y_offset):
        result[y_offset: y_offset + display.shape[0], x_offset: x_offset + display.shape[1]] = display
        return result
