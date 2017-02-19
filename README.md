**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_not_car.jpg
[image2]: ./output_images/hog_features.jpg
[image3]: ./output_images/sliding_windows140.jpg
[image4]: ./output_images/sliding_windows120.jpg
[image5]: ./output_images/sliding_windows96.jpg
[image6]: ./output_images/sliding_windows64.jpg
[image7]: ./output_images/sliding_window.jpg
[image8]: ./output_images/bboxes_and_heat.png
[image9]: ./output_images/labels_map.png
[video1]: ./output_images/project_video_output.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in lines 5 through 13 of the file called `features.py`.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and based on the svm training results I chose the ones that were maximizing the validation set performance.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using the hog features from the image in combination of color histogram and the resized image. I split the data into training and validation sets
 80% training and 20% validation. I also trained a scaler to normalize the data before training the SVM model.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search the image starting from 400 pixels to 650. I had multiple size search windows.
140px, 0.5 overlap, 400-650 height
120px, 0.7 overlap, 400-600 height
96px, 0.7 overlap, 400 - 550 height
64px, 0.7 overlap , 400-500 height

This is example of the 140px sliding wind
![alt text][image3]

This is example of the 120px sliding wind
![alt text][image4]

This is example of the 96px sliding wind
![alt text][image5]

This is example of the 64px sliding wind
![alt text][image6]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using  LUV 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.
Here are some example images:

![alt text][image7]
-------------------

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output_images/project_video_output.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image8]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image9]




---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The implementation problems I faced is in getting the pipeline to perform sable and robustly. I had to experiment with many window sizes and overlaps until I got a stable implementation.
Also the time to produce the video was quite large. The pipeline is likely to fail if the detection is happening at night since the provided features may no longer work.
To make it more robust I think a better search algorithm can be used to speed things up. I also think that convolutional neural network will be much better at detecting cars compared to
the hand crafted feature vectors used.