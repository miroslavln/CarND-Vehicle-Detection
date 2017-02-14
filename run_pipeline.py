import glob
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from features import extract_features, load_image
from vehicle_detector import VehicleDetector
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip

if __name__ == "__main__":
    cars = glob.glob('data/vehicles/*/*')
    noncars = glob.glob('data/non-vehicles/*/*')

    car_features = extract_features(cars)
    noncar_features = extract_features(noncars)
    X = np.vstack((car_features, noncar_features)).astype(np.float64)
    print(X.shape)

    car_labels = np.ones(len(cars))
    noncar_labels = np.zeros(len(noncars))
    y = np.hstack((car_labels, noncar_labels))
    print(y.shape)

    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

    svc = LinearSVC()
    t = time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()

    acc = svc.score(X_test, y_test)
    print('Accuracy', acc)

    test_image = load_image('test_images/test4.jpg')

    detector = VehicleDetector(svc, scaler, 0.7)
    res = detector.process_image(test_image)

    plt.imshow(res)
    plt.show()

    '''

    detector = VehicleDetector(svc, scaler, 0.7)
    white_output = './output_images/project_video_output.mp4'
    clip1 = VideoFileClip("./project_video.mp4")
    white_clip = clip1.fl_image(detector.process_image)
    white_clip.write_videofile(white_output, audio=False)

    '''