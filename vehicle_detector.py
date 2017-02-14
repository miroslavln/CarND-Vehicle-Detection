import glob
import time

import numpy as np
import cv2
from features import get_features
from scipy.ndimage.measurements import label

class VehicleDetector:
    def __init__(self, clf, scaler, overlap):
        self.heatmap = None
        self.sizes = [100]
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
    def get_windows(img, y_start_stop, window_size, overlap=0.7):
        h, w, c = img.shape
        xspan = w
        yspan = y_start_stop[1] - y_start_stop[0]

        step = int(window_size * overlap)
        x_windows = (xspan // step) - 1
        y_windows = (yspan // step) - 1

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
        features_list = []
        detected = []
        for window in windows:
            x1, y1 = window[0]
            x2, y2 = window[1]
            test_img = img[y1:y2, x1:x2, :]
            assert test_img is not None
            test_img = cv2.resize(test_img, (64, 64))
            features = get_features(test_img)
            features_list.append(features)
            features = self.scaler.transform(features.reshape(1,-1))
            prediction = self.clf.predict(features)
            if prediction == 1:
                detected.append(window)

        return detected
        features_list = np.array(features_list)
        features_list = self.scaler.transform(features_list)
        prediction = self.clf.predict(features_list)

        return [windows[i] for i in range(len(windows)) if prediction[i] == 1]

    def add_heat(self, bboxes):
        for box in bboxes:
            x1, y1 = box[0]
            x2, y2 = box[1]
            self.heatmap[y1:y2, x1:x2] += 1

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
        #if self.heatmap is None:
        #    self.heatmap = np.zeros_like(img[:, :, 0])

        heatmap = np.zeros_like(img[:, :, 0]).astype(np.float32)
        heatmap = np.clip(heatmap, 0, 255)

        #self.heatmap -= 5
        #self.heatmap = np.clip(self.heatmap, 0, 255)

        h, w, c = img.shape
        height = h // 2
        for i, size in enumerate(self.sizes):
            y_start_stop = (height, height + height // (i + 1))

            windows = self.get_windows(img, y_start_stop, size, self.overlap)
            detected = self.search_windows(img, windows)
            self.add_heat(detected)

        self.apply_threashold(2)
        labels = label(self.heatmap)
        bboxes = self.get_labeled_boxes(labels)

        return self.draw_boxes(img, bboxes)

