
## Writeup Template
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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
[image1]: ./output_images/Car.jpg
[image2]: ./output_images/Non-car.jpg
[image3]: ./output_images/Hog.jpg
[image4]: ./output_images/hotwindows.jpg
[image5]: ./output_images/heat.jpg
[video1]: ./project_video_output.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the code cell #10 of the IPython notebook.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here are a few of the `vehicle` and `non-vehicle` classes:

Vehicles:
![alt text][image1]

Non-vehicles:
![alt text][image2]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like. Along with the HOG features, I also used spatial and color histogram features shown in cell #11 of the notebook.

Here is an example using the `RGB` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image3]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters for the color space and the HOG features. Then I trained a LinearSVC classifier for each combination as shown in the table below.

| colorspace| orient|pix_per_cell|cell_per_block|HOG channel| accuracy|  time  |
|:---------:|:-----:|:----------:|:------------:|:---------:|:-------:|:------:|
| RGB | 9 | 8  | 2  | ALL  | 0.975 | 42secs |
| LUV | 9 | 8  | 2  | ALL  | 0.977 | 28secs |
| HSV | 9 | 8  | 2  | ALL  | 0.977 | 26secs |
| HLS | 9 | 8  | 2  | ALL  | 0.975 | 27secs |
| YUV | 9 | 8  | 2  | ALL  | 0.975 | 27secs |
| YCrCb | 9 | 8  | 2  | ALL  | 0.98 | 26secs |
| YCrCb | 11 | 16  | 2  | ALL  | 0.974 | 13secs |
| YUV | 11 | 16  | 2  | ALL  | 0.972 | 13secs |
| HLS | 11 | 16  | 2  | ALL  | 0.976 | 13secs |
| HSV | 11 | 16  | 2  | ALL  | 0.977 | 14secs |
| LUV | 11 | 16  | 2  | ALL  | 0.971 | 15secs |
| RGB | 11 | 16  | 2  | ALL  | 0.976 | 22secs |

I selected YCrCb colorspace with Orient=9, pix_per_cell=8,cell_per_block=2 and HOG='ALL'
for the final model since it had the highest accuracy. 

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM (in cell #14) using color and HOG features. I converted RGB colored image to YCrCb, and computed its spatial features, histogram features and HOG features (in cell #13). It was then applied to the car and noncar images and the extracted features were trained using LinearSVC classifier. The accuracy of the classifer was 98.9%.


### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The sliding window search is located in cells #16, #17, #18.

The search area of the sliding windows were restricted to bottom-half (to avoid the sky) and a little to the right (to avoid detecting opposite side cars) of the image. This was achieved by using y_start_stop and x_start_stop variables.

I used three combinations of window scales and tried overlap of 0.6,0.65 and 0.7. My final pipeline had  three combinations of window scales((64,64),(96,96),(128,128)) and an overlap of 0.7.


#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on three scales ((64,64),(96,96),(128,128)) using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result. This helped the classifier to perform well.  Here are some example images:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_output.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  This is in cell #46 of the notebook. 

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames, their corresponding heatmaps and resulting bounding boxes drawn:

![alt text][image5]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

One big problem I faced was the time it took to run the video. When problems like false positives were found, it was a slow process to fix, run the video and check if it was fixed.

My pipeline will likely fail to detect a pedestrian, motorcycle etc. I could use more training images to make it more robust.



