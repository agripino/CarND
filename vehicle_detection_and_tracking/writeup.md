##Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_sample.png
[image2]: ./output_images/noncar_sample.png
[image3]: ./output_images/car_hog_vis_ch1.png
[image4]: ./output_images/car_hog_vis_ch2.png
[image5]: ./output_images/car_hog_vis_ch3.png
[image6]: ./output_images/noncar_hog_vis_ch1.png
[image7]: ./output_images/noncar_hog_vis_ch2.png
[image8]: ./output_images/noncar_hog_vis_ch3.png
[image9]: ./output_images/search_window.png
[image10]: ./output_images/sample_pipeline1.png
[image11]: ./output_images/sample_pipeline2.png
[image12]: ./output_images/sample_pipeline3.png
[image13]: ./output_images/heatmap0.png
[image14]: ./output_images/heatmap1.png
[image15]: ./output_images/heatmap2.png
[image16]: ./output_images/heatmap3.png
[image17]: ./output_images/heatmap4.png
[image18]: ./output_images/heatmap5.png
[image19]: ./output_images/labels.png
[image20]: ./output_images/bounding_boxes.png
[video11]: ./videos/project_video_annotated.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the function `get_preprocessed_data` in the file `classifier_training.py` (lines
15 to 61). That function calls the function `extract_features`, defined in `features.py` (lines 47 to 57).

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle`
and `non-vehicle` classes:

![alt text][image1]
![alt text][image2]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and
`cells_per_block`).

Here is an example of the original images and HOG visualizations of each channel using the `YCrCb` color space and HOG
parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image1]
![alt text][image3]
![alt text][image4]
![alt text][image5]

![alt text][image2]
![alt text][image6]
![alt text][image7]
![alt text][image8]


####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters taking this video (https://www.youtube.com/watch?v=7S5qXET179I) as a reference.
Using the default ones (`orientations=9, pix_per_cell=8, cell_per_block=2`) and histogram features in the HLS color space
resulted in a classifier with good performance.

Decreasing the number of orientations can reduce the number of features
significantly, which means faster training and classification. However, a small number of orientations does not provide
enough capacity to distinguish objects effectively.

Since HOG features describe the shape of objects, it was applied to a gray scale image, while the color aspect was
was captured using histograms.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a SVM classifier using HOG features of a gray scale image and color histograms in the HLS color space. The
code is in the function `train_classifier` (file `classifier_training.py`, lines 64 to 84). The standard scaler is
created and pickled in the `get_preprocessed_data` function (lines 15 to 61 of the same file). It's used later by
the `find_cars` function.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The sliding window search is implemented in the `find_cars` function (lines 43 to 99 of the file `vehicle_detection.py`).
Because of the long time required to run the search, a single scale was chosen based on the average dimensions of cars
near the vehicle. An overlap of 2 cells (32 pixels in the original image with a scale of 2) was chosen as a trade-off
between execution time and how tight the boxes wrapped detected cars. 

![alt text][image9]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on a single scale using gray scale HOG features plus histograms of color in the HLS color space.

Detections were accumulated over 10 frames and a heatmap was used to improve the performance of the pipeline. At least 2
detections were required to make a pixel "hot" enough to appear in the final detections.

Performance was also improved by the choice of complementary features: HOG features for shape and histogram features for
color.

Here are some example images:

![alt text][image10]
![alt text][image11]
![alt text][image12]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./videos/project_video_annotated.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections for each search region and for each frame along 10 frames in the
integrated detections variable (line 11 of the the file `vehicle_detection.py`).

From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions (lines 117
to 123 of `vehicle_detection.py`). I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the
heatmap (line 126 of `vehicle_detection.py`). I then assumed each blob corresponded to a vehicle. I constructed bounding
boxes to cover the area of each blob detected (line 128 of the same file).

Here's an example result showing the heatmap from a series of frames of video, the result of
`scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image13]
![alt text][image14]
![alt text][image15]
![alt text][image16]
![alt text][image17]
![alt text][image18]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image19]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image20]

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The main problem faced during implementation was the time required to run the pipeline, even if multiprocessing was used.
That led me to use a single search region.

In order to get tighter boxes wrapping the cars, it would be necessary to use smaller scales combined with smaller, more
selective search regions, also limited in the x axis, for example.

Variable light conditions may affect the pipeline negatively. A square root transform could be included during HOG
feature extraction in order to tackle this problem. Here it was not used in an attempt to speed up execution time.

The pipeline could run on each search region using a different thread, and good choices of region limits and scales would
distribute the computational load evenly among threads.

To increase robustness I would also take a more systematic approach to define the features and the classification method used
so as to obtain a better classifier.
